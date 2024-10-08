#!/usr/bin/env python
""" collections of utility functions """

import sys
import os
import csv
import numpy as np
import scipy.io.wavfile


class Instance(object):
    """
        Instance class represents set of raw data collected per data instance
    """
    def __init__(self, dir):
        self.accel = self._load_accel(dir)
        self.touch = self._load_touch_data(dir)
        self.info = self._load_instance_info(dir)

    def _load_accel(self, dir):
        """ load accel data
            param dir: a path to an instance directory
            return: accel data
        """
        rate, wav = scipy.io.wavfile.read(os.path.join(dir, "accel.wav"))
        return wav

    def _load_touch_data(self, dir):
        """ load touch data
            param dir: a path to an instance directory
            return : a dictionary contains touch data
        """
        with open(os.path.join(dir, "touch.csv"), "rU") as f:
            reader = csv.DictReader(f)
            for touch in reader:
                for key in touch.keys():
                    touch[key] = float(touch[key])
                break
        return touch

    def _load_instance_info(self, dir):
        """ load instance info from a directory path
            param dir: a path to an instance directory
            return: a dictionary contains basic instance information
        """
        info = {}
        user_dirnames = os.path.basename(os.path.dirname(dir)).split("-")
        info["surface"] = user_dirnames[0]
        info["user"] = user_dirnames[1]
        instance_dirnames = os.path.basename(dir).split("-")
        info["timestamp"] = instance_dirnames[0]
        # set None to classlabel if it's test data
        info["classlabel"] = instance_dirnames[1] if len(instance_dirnames) == 2 else None
        return info


def load_instances(dir):
    """ function for loading raw data instances
        param dir: a path to a data directory (i.e. task_data/train or task_data/test)
        return: a list of data instance objects
    """
    instances = []
    for root, dirs, files in os.walk(os.path.join(dir)):
        for filename in files:
            if filename == "accel.wav":
                instances.append(Instance(root))
    return instances


def load_labels(instances):
    """ load class labels
        param instances: a list of data instance objects
        return: class labels mapped to a number (0=pad, 1=knuckle)
    """
    y = np.array([{"pad": 0, "knuckle": 1}[instance.info["classlabel"]] for instance in instances], dtype=int)
    return y


def load_timestamps(instances):
    """ load timestamps
        param instances: a list of data instance objects
    """
    timestamps = [instance.info["timestamp"] for instance in instances]
    return timestamps


def convert_to_classlabels(y):
    """ convert to classlabels
        param y: mapped class labels
        return: class labels
    """
    classlabels = [["pad", "knuckle"][y[i]] for i in range(len(y))]
    return classlabels


def write_results(timestamps, classlabels, output):
    """ write classification results to an output file
        param timestamps: a list of timestamps
        param classlabels: a list of predicted class labels
        return : None
    """
    if len(timestamps) != len(classlabels):
        raise Exception("The number of timestamps and classlabels doesn't match.")
    with open(output, "w") as f:
        f.write("timestamp,label\n")
        for timestamp, classlabel in zip(timestamps, classlabels):
            f.write(timestamp + "," + classlabel + "\n")


def main(argv):
    raise Exception("This script isn't meant to be run.")


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
