import json
import os
from analysis_train import InstanceData


def extract_speed_from_result(result_dir: str):
    with open(os.path.join(result_dir, "results.json"), "r") as f:
        instance_data = [
            InstanceData(**i)
            for i in json.load(f)
        ]
    overall_data = instance_data[-1]
    return overall_data.token_per_second


def gen_result_table(result_root: str):
    for clients in [1, 2, 4, 8]:
        for client_config in \
                [
                    "_p_4_8", "_p_4_32",
                    "_p_16_8", "_p_16_32",
                    "_r_4_8", "_r_4_32",
                    "_r_16_8", "_r_16_32"
                ]:
            output_dir = f"{result_root}/{clients}{client_config}"
            speed = extract_speed_from_result(output_dir)
            print(f"& {speed:>6.2f}", end="")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    args, _ = parser.parse_known_args()
    output_dir = args.output
    gen_result_table(output_dir)
