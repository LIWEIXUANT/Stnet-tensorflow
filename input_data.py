import os
import time
import threading
import queue
import random
import numpy as np
from numpy.random import randint
import cv2
import glob


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class GeneratorEnqueuer(object):
    def __init__(self, workers,
                       root_path,
                       file_list,
                       input_size,
                       num_segment,
                       batch_size,
                       shuffle,
                       prefix,
                       transform):
        self.workers = workers
        self.root_path = root_path
        self.file_list = file_list
        self.input_size = input_size
        self.num_segment = num_segment
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_tmpl = prefix
        self.transform = transform
        self.queue = queue.Queue(maxsize=10)
        self._threads = []

        self._parse_list()
        self.total_list = self.video_list


    def start(self):
        try:
            video_number = len(self.video_list)
            if self.shuffle:
                random.shuffle(self.video_list)

            sub_file_num = int(video_number / self.workers)
            for i in range(self.workers):
                if i != self.workers - 1:
                    sub_file_list = self.video_list[i*sub_file_num: (i+1)*sub_file_num]
                else:
                    sub_file_list = self.video_list[i*sub_file_num:]
                thread = threading.Thread(target=self.task, args=(sub_file_list, ))
                thread.daemon = True
                self._threads.append(thread)
                thread.start()

        except:
            raise

    def _sample_indices(self, record):
        average_duration = record.num_frames // self.num_segment
        offsets = np.multiply(list(range(self.num_segment)), average_duration) + randint(average_duration, size=self.num_segment)
        return offsets + 1

    def task(self, sub_file_list):
        batch_images = []
        batch_labels = []
        #print(sub_file_list)
        for record in sub_file_list:
            segment_indices = self._sample_indices(record)
            images, label = self.get(record, segment_indices)
            batch_images.extend(images)
            batch_labels.append(label)
            #print('//////////////////////////////////')
            #print(len(batch_images))
            #print(len(batch_labels))
            if len(batch_labels) == self.batch_size:
                self.queue.put((batch_images, batch_labels))
                batch_images = []
                batch_labels = []

    def get(self, record, indices):
        #print(record.path)
        images = [self._load_image(record.path, int(idx)) for idx in indices]
        data = self.transform(images)
        label = record.label
        return data, label

    def _load_image(self, directory, idx):
        #print(os.path.join(self.root_path, directory, directory + '-' + self.image_tmpl.format(idx)))
        image = cv2.imread(os.path.join(self.root_path, directory, directory + '-' + self.image_tmpl.format(idx)))[:, :,::-1]
        return image

    def is_running(self):
        return any([th.is_alive() for th in self._threads])


    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.file_list)]
        tmp = [item for item in tmp if int(item[1]) >= self.num_segment]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number: {}'.format(len(self.video_list)))

    def get_negatives(self, sess, params, sample_numbers):
        positive_samples = [obj for obj in self.total_list if obj.label == 1]
        negative_samples = [obj for obj in self.total_list if obj.label == 0]
        losses = []
        if sample_numbers is None:
            sample_numbers = len(positive_samples)

        for i, sample in enumerate(negative_samples):
            try:
                frame_paths = sorted(glob.glob(os.path.join(self.root_path, sample.path, '*.jpg')))
                frames = load_frames(frame_paths, num_frames=self.num_segment)
                data = self.transform(frames)
                loss = sess.run(params['loss_'], feed_dict={params['x_']: data, params['y_']: [sample.label]})

                losses.append(loss)

            except Exception as e:
                pass

            if i % 100 == 0:
                print('\r>>> {}/{} processing, average loss: {:.5}'.format(i, len(negative_samples), sum(losses)/len(losses)), end='')

        idex = np.argsort(-1 * np.array(losses))
        select_idex = idex[:sample_numbers]
        hard_negative_samples = [negative_samples[i] for i in select_idex]
        time_tample = str(time.time())
        with open('./log/hard_sample_{}.txt'.format(time_tample), 'w') as f:
            f.write('\n'.join([obj.path for obj in hard_negative_samples]))
        self.video_list = positive_samples + hard_negative_samples
        random.shuffle(self.video_list)
        print('\nSelect hard negative samples done!')

def load_frames(frame_paths, num_frames=8):
    if len(frame_paths) > num_frames:
        frame_paths = frame_paths[::int(np.floor(len(frame_paths) / float(num_frames)))][:num_frames]
        return [cv2.imread(frame) for frame in frame_paths]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))

class DataLoader(object):
    def __init__(self, **kwargs):
        self.enqueuer = GeneratorEnqueuer(**kwargs)
        print('Create queue sucessing!')

    def run(self):
        self.enqueuer.start()
        generator_output = None
        while True:
            if not self.enqueuer.queue.empty():
                generator_output = self.enqueuer.queue.get()
                yield generator_output
                generator_output = None
            else:
                if self.enqueuer.is_running():
                    time.sleep(0.01)
                else:
                    yield generator_output

    def select_harder_negatives(self, sess, params, sample_numbers=None):
        self.enqueuer.get_negatives(sess, params, sample_numbers)


    def __iter__(self):
        return self

    def __len__(self):
        return len(self.enqueuer.video_list) // self.enqueuer.batch_size
