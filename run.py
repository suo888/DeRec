proj_path = "/home..../" # set the absolute path of this project

import sys, argparse
sys.path.append(proj_path)

from Runs.rank_task import Run
from configparser import ConfigParser
cfg = ConfigParser()


import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name')
    parser.add_argument('--model_name', nargs='?', help='model name')
    parser.add_argument('--timestamp', default=None, nargs='?', help='timestamp')

    args = parser.parse_args()
    data = args.data_name
    model = args.model_name
    timestamp = args.timestamp

    if timestamp == None:
        mode = 'train'
    else:
        mode = 'test'
    print("34455")
    # ======= get the running setting ========
    cfg.read(".../R/Runs/configurations/example/example_final.ini")
    print(2)
    print(model)
    # ======= run the main file ============
    # cfg.items：读取配置文件
    Run(DataSettings   = dict(cfg.items("DataSettings")),
        ModelSettings  = dict(cfg.items("ModelSettings")),
        TrainSettings  = dict(cfg.items("TrainSettings")),
        ResultSettings = dict(cfg.items("ResultSettings")),
        mode=mode,
        timestamp=timestamp
        )

