import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


class HungarianMatcher(nn.Module):
    """Standard Hungarian matcher for initial task (task 0)."""

    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class QMcMfMatcher(nn.Module):
    """Quality-guided Min-Cost Max-Flow matcher for incremental tasks.

    Implements the method described in the paper:
    1) Build bipartite graph: predictions <-> targets
    2) Prune edges with IoU below threshold τ(q_j) — geometrically implausible matches removed
    3) Solve min-cost max-flow on the sparse graph with one-to-one capacity
    4) A high-cost source-to-sink edge absorbs unmatched flow
    5) Matching maximizes valid assignments while minimizing total cost

    Uses OR-Tools SimpleMinCostFlow (integer costs with precision=100000).
    """

    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1,
                 iou_threshold_gt=0.5, iou_threshold_pseudo=0.7,
                 max_flow=300):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.iou_threshold_gt = iou_threshold_gt
        self.iou_threshold_pseudo = iou_threshold_pseudo
        self.max_flow = max_flow
        self.precision = 100000
        self.unmatched_cost = 9999 * self.precision
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """Match predictions to targets.

        targets[i] may contain:
          - "boxes", "labels": ground-truth annotations
          - "is_pseudo" (optional, bool mask): True for teacher-generated pseudo-labels
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1)

            sizes = [len(v["boxes"]) for v in targets]

            indices = []
            for i in range(bs):
                num_tgt = sizes[i]
                cost_mat = C[i, :, :num_tgt]

                # Compute IoU (not GIoU) for quality-guided edge pruning, as per paper
                iou, _ = box_iou(
                    box_cxcywh_to_xyxy(out_bbox[i * num_queries:(i + 1) * num_queries]),
                    box_cxcywh_to_xyxy(tgt_bbox[sum(sizes[:i]):sum(sizes[:i + 1])])
                )

                # Per-target threshold: pseudo-label vs ground-truth
                per_target_thresh = self.iou_threshold_gt
                if "is_pseudo" in targets[i]:
                    per_target_thresh = torch.where(
                        targets[i]["is_pseudo"],
                        self.iou_threshold_pseudo,
                        self.iou_threshold_gt
                    )

                src_idx_list, tgt_idx_list = self._solve_mcmf(cost_mat, iou, per_target_thresh)
                indices.append((
                    torch.as_tensor(src_idx_list, dtype=torch.int64),
                    torch.as_tensor(tgt_idx_list, dtype=torch.int64)
                ))

            return indices

    def _solve_mcmf(self, cost_mat, iou_mat, per_target_thresh):
        P, Q = cost_mat.shape
        if P == 0 or Q == 0:
            return [], []

        from ortools.graph.python import min_cost_flow

        start_nodes = []
        end_nodes = []
        capacities = []
        unit_costs = []

        source = 0
        sink = 1
        n_nodes = 2 + P + Q

        # Source -> sink direct edge: absorbs unmatched flow at high cost
        start_nodes.append(source)
        end_nodes.append(sink)
        capacities.append(self.max_flow)
        unit_costs.append(self.unmatched_cost)

        # Source -> prediction edges (capacity 1, cost 0)
        for i in range(P):
            start_nodes.append(source)
            end_nodes.append(2 + i)
            capacities.append(1)
            unit_costs.append(0)

        # Prediction -> target edges
        # Only keep edges with IoU >= threshold for the target (quality-guided edge pruning)
        # This is the core of Q-MCMF: eliminates geometrically implausible matches
        edge_start = len(start_nodes)
        for i in range(P):
            for j in range(Q):
                start_nodes.append(2 + i)
                end_nodes.append(2 + P + j)
                if iou_mat[i, j].item() >= (per_target_thresh[j].item() if hasattr(per_target_thresh, '__len__') else per_target_thresh):
                    # Plausible match: use quantized matching cost
                    c = int(round(cost_mat[i, j].item() * self.precision))
                    unit_costs.append(max(c, 1))
                    capacities.append(1)
                else:
                    # Implausible match: prune the edge (zero capacity)
                    capacities.append(0)
                    unit_costs.append(0)

        # Target -> sink edges (capacity 1, cost 0)
        for j in range(Q):
            start_nodes.append(2 + P + j)
            end_nodes.append(sink)
            capacities.append(1)
            unit_costs.append(0)

        smcf = min_cost_flow.SimpleMinCostFlow()
        arc_indices = smcf.add_arcs_with_capacity_and_unit_cost(
            start_nodes, end_nodes, capacities, unit_costs)

        supplies = [self.max_flow] + [-self.max_flow] + [0] * (P + Q)
        smcf.set_nodes_supplies(list(range(n_nodes)), supplies)

        status = smcf.solve()
        if status != smcf.OPTIMAL:
            return [], []

        src_idx_list = []
        tgt_idx_list = []
        for i in range(P):
            for j in range(Q):
                arc_idx = edge_start + i * Q + j
                if smcf.flow(arc_idx) > 0:
                    src_idx_list.append(i)
                    tgt_idx_list.append(j)

        return src_idx_list, tgt_idx_list


def build_matcher(args):
    if args.use_qmcmf:
        return QMcMfMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            iou_threshold_gt=args.qmcmf_iou_thresh,
                            iou_threshold_pseudo=args.qmcmf_pseudo_iou_thresh)
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
