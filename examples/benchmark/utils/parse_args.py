import argparse


def parse_args(model_name, device):
    parser = argparse.ArgumentParser(
        description=f"Benchmark {model_name} models on {device}. The script can be to be launched in parallel to a "
                    f"{device} system management too in order to collect additional resource consumption data."
    )
    parser.add_argument(
        "--log_file", help="file path of the running logs.", type=str, required=True
    )
    parser.add_argument(
        "--sleep_interval", help="sleep interval in seconds before two inference sessions", type=int, default=4
    )
    parser.add_argument(
        "--run_interval", help="target running time in seconds", type=int, default=4
    )

    # parse arguments
    args = parser.parse_args()
    log_file = args.log_file
    sleep_interval = args.sleep_interval
    run_interval = args.run_interval
    return log_file, sleep_interval, run_interval
