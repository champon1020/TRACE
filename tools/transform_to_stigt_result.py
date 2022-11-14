import argparse
import json
import os.path as op
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--rel_detections_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)


class DataEnconding(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
                obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, )):  # add this line
            return obj.tolist()  # add this line
        elif isinstance(obj, (torch.Tensor, )):
            return obj.cpu().detach().tolist()
        return json.JSONEncoder.default(self, obj)


def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(
        np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def to_image_id(image_path: str):
    return "/".join(image_path.split("/")[-2:])


def main(args):
    results_targets = {}
    dets = pickle.load(open(args.rel_detections_file, "rb"))

    topk, prdk = 100, 2

    for i, det in enumerate(dets):
        image_idx = i
        image_id = to_image_id(det["image"])
        image_id = op.splitext(image_id)[0] + ".jpg"

        # Process predictions
        det_boxes_sbj = det["sbj_boxes"]  # (#num_rel, 4)
        det_boxes_obj = det["obj_boxes"]  # (#num_rel, 4)
        det_labels_sbj = det["sbj_labels"]  # (#num_rel,)
        det_labels_obj = det["obj_labels"]  # (#num_rel,)
        det_scores_sbj = det["sbj_scores"]  # (#num_rel,)
        det_scores_obj = det["obj_scores"]  # (#num_rel,)
        det_scores_prd = det["prd_scores_ttl"]  # (#num_rel, n_prd_classes)

        det_labels_prd = np.argsort(-det_scores_prd, axis=1)
        det_scores_prd = -np.sort(-det_scores_prd, axis=1)
        det_scores_so = det_scores_sbj * det_scores_obj
        det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :prdk]

        det_scores_inds = argsort_desc(det_scores_spo)[:topk]
        det_scores_top = det_scores_spo[det_scores_inds[:, 0],
                                        det_scores_inds[:, 1]]

        # >> original
        det_scores_s_top = det_scores_sbj[det_scores_inds[:, 0]]
        det_scores_o_top = det_scores_obj[det_scores_inds[:, 0]]
        # << original

        det_boxes_so_top = np.hstack((det_boxes_sbj[det_scores_inds[:, 0]],
                                      det_boxes_obj[det_scores_inds[:, 0]]))
        det_labels_p_top = det_labels_prd[det_scores_inds[:, 0],
                                          det_scores_inds[:, 1]]
        det_labels_spo_top = np.vstack((
            det_labels_sbj[det_scores_inds[:, 0]],
            det_labels_p_top,
            det_labels_obj[det_scores_inds[:, 0]],
        )).transpose()

        # filter out bad relationships
        cand_inds = np.where(det_scores_top > 0.00001)[0]
        det_boxes_so_top = det_boxes_so_top[cand_inds]
        det_labels_spo_top = det_labels_spo_top[cand_inds]
        det_scores_top = det_scores_top[cand_inds]
        det_scores_s_top = det_scores_s_top[cand_inds]
        det_scores_o_top = det_scores_o_top[cand_inds]

        # det_scores_vis = det['prd_scores']
        # for i in range(det_labels_prd.shape[0]):
        #     det_scores_vis[i] = det_scores_vis[i][det_labels_prd[i]]
        # det_scores_vis = det_scores_vis[:, :prdk]
        # det_scores_top_vis = det_scores_vis[det_scores_inds[:, 0],
        #                                     det_scores_inds[:, 1]]
        # det_scores_top_vis = det_scores_top_vis[cand_inds]

        det_boxes_s_top = det_boxes_so_top[:, :4]
        det_boxes_o_top = det_boxes_so_top[:, 4:]
        det_labels_s_top = det_labels_spo_top[:, 0]
        det_labels_p_top = det_labels_spo_top[:, 1]
        det_labels_o_top = det_labels_spo_top[:, 2]

        num_rel = det_labels_p_top.shape[0]

        rels = []
        for i in range(num_rel):
            rels.append([i, num_rel + i, det_labels_p_top[i]])
        if len(rels) == 0:
            continue
        boxes = np.concatenate([det_boxes_s_top, det_boxes_o_top],
                               axis=0).tolist()
        labels = np.concatenate([det_labels_s_top, det_labels_o_top],
                                axis=0).tolist()
        scores = np.concatenate([det_scores_s_top, det_scores_o_top],
                                axis=0).tolist()
        rel_scores = det_scores_top.tolist()

        # Process GT
        gt_sbj_boxes = det["gt_sbj_boxes"]
        gt_obj_boxes = det["gt_obj_boxes"]
        gt_sbj_labels = det["gt_sbj_labels"]
        gt_obj_labels = det["gt_obj_labels"]
        gt_prd_labels = det["gt_prd_labels"]

        num_rel = gt_prd_labels.shape[0]
        gt_rels = []
        for i in range(num_rel):
            gt_rels.append([i, num_rel + i, gt_prd_labels[i]])
        gt_boxes = np.concatenate([gt_sbj_boxes, gt_obj_boxes], axis=0)
        gt_labels = np.concatenate([gt_sbj_labels, gt_obj_labels], axis=0)

        results_targets[image_id] = {
            "result": {
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "rels": rels,
                "rel_scores": rel_scores,
            },
            "target": {
                "boxes": gt_boxes,
                "labels": gt_labels,
                "rels": gt_rels,
                "image_idx": image_idx,
            },
        }

    output_path = op.join(args.output_dir, "results.json")
    with open(output_path, "w") as fp:
        json.dump(results_targets, fp, cls=DataEnconding)
    print(f"save results to {output_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
