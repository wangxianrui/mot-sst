import os
import argparse
from config import Config


def main(args):
    result_dir = os.path.join(Config.data_root, 'detection', args.type)
    video_list = os.listdir(os.path.join(Config.data_root, args.type))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for video_name in video_list:
        command = 'cp {} {}'.format(os.path.join(Config.data_root, args.type, video_name, 'det/det.txt'),
                                    os.path.join(result_dir, video_name + '.txt'))
        print(command)
        os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='train')
    args = parser.parse_args()
    main(args)
