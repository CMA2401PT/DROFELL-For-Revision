import json
import os
from typing import (Dict, List, Tuple)


def compute_avg_speed(record: List[Tuple[int, int]], range: int) -> float:
    c: Dict[int, int] = {}
    for (i, end_t) in record:
        if not i in c:
            c[i] = 1
        else:
            c[i] += 1
        if min(c.values()) == range:
            return (sum(c.values())/((end_t-record[0][1])/(10**9)))
    raise ValueError("data insufficient")


def extract_speed_from_result(result_file: str, range: int):
    with open(result_file, "r") as f:
        result = json.load(f)
    return compute_avg_speed(result, range)


def gen_result_table(result_root: str):
    for devide_config in \
            [
                "1_1_1", "1_2_1", "1_4_1",
                "2_2_1", "2_4_1",
                "2_1_2", "2_2_2", "2_4_2"
            ]:
        for range_len in [128, 1024]:
            for drop_prob_name in ["drop_p0", "drop_p1e-4", "drop_p1e-3", "drop_p1e-2", "drop_p5e-2"]:
                speeds = []
                for repeat in range(10):
                    output_dir = f"{result_root}/{devide_config}/{drop_prob_name}/repeat_{repeat}"
                    s = extract_speed_from_result(os.path.join(output_dir, "result.json"), range_len)
                    speeds.append(s)
                print(f"& {sum(speeds)/len(speeds):>6.2f}", end="")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    args, _ = parser.parse_known_args()
    output_dir = args.output
    gen_result_table(output_dir)
