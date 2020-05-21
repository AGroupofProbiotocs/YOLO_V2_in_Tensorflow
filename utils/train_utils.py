import numpy as np

class Save_Manager():
    '''
    Managing the modeling saving during training, which supports monitoring of multiple metrics.
    '''
    def __init__(self, saver, ckpt_path, patient, metrics, mode=None, max_metrics=np.inf,
                 min_metrics=-np.inf, save_best=True, early_stop=True, epsilon=1e-5):
        self.saver = saver
        self.ckpt_path = ckpt_path
        self.patient = patient
        self.metrics = metrics
        self.mode = mode
        self.max_metrics = max_metrics
        self.min_metrics = min_metrics
        self.save_best = save_best
        self.early_stop = early_stop
        self.epsilon = epsilon
        self.variable_dict = {'stop_training':False}
        self.dict_templete = {'min_value':self.max_metrics,
                              'max_value':self.min_metrics,
                              'wait':0,
                              'stoppable':False,
                              'epoch':0,
                              'mode':None}

        if type(self.metrics) == str:
            self.metrics = [metrics]
        if type(self.metrics) == list:
            if self.mode is not None:
                if type(self.mode) == str:
                    self.mode = [mode]
                elif type(self.mode) == list:
                    if len(self.mode)!=len(self.metrics):
                        raise ValueError('Metrics and mode should be with same length!')
                else:
                    raise ValueError('Unacceptable mode type!')
            else:
                self.mode = [None]*len(self.metrics)

            for md, mt in zip(self.mode, self.metrics):
                self.variable_dict[mt] = self.dict_templete
                self.variable_dict[mt]['mode'] = md
        else:
            raise ValueError('Unacceptable metrics type!')


    def save_decision(self, session, ckpt_path, metrics, current_value, mode, current_epoch):
        if mode is None:
            if metrics.split(' ')[1] == 'loss':
                mode = 'min'
                self.variable_dict[metrics]['mode'] = 'min'
            elif metrics.split(' ')[1] == 'accuracy':
                mode = 'max'
                self.variable_dict[metrics]['mode'] = 'max'
            else:
                raise ValueError('Not predefined metrics! The mode should be set to be "max" or "min"!')

        a, b = metrics.split(' ')
        ckpt_path = ckpt_path + '_' + mode + '_' + a + '_' + b + '.ckpt'

        if mode == 'min':
            if current_value < self.variable_dict[metrics]['min_value'] - self.epsilon:
                print('%s decreased from %.5f to %.5f' % (metrics, self.variable_dict[metrics]['min_value'], current_value))
                self.variable_dict[metrics]['min_value'] = current_value
                self.variable_dict[metrics]['epoch'] = current_epoch
                if self.save_best:
                    self.saver.save(session, ckpt_path)
                    print("Model saved in file: %s" % ckpt_path)
                if self.early_stop:
                    self.variable_dict[metrics]['wait'] = 0
            else:
                print('%s did not decreased.'% (metrics))
                if self.early_stop:
                    self.variable_dict[metrics]['wait'] += 1
                    if self.variable_dict[metrics]['wait'] > self.patient:
                        print('Early stop!')
                        self.variable_dict[metrics]['stoppable'] = True

        elif mode == 'max':
            if current_value > self.variable_dict[metrics]['max_value'] + self.epsilon:
                print('%s increased from %.5f to %.5f' % (metrics, self.variable_dict[metrics]['max_value'], current_value))
                self.variable_dict[metrics]['max_value'] = current_value
                self.variable_dict[metrics]['epoch'] = current_epoch
                if self.save_best:
                    self.saver.save(session, ckpt_path)
                    print("Model saved in file: %s" % ckpt_path)
                if self.early_stop:
                    self.variable_dict[metrics]['wait'] = 0
            else:
                print('%s did not increased.'% (metrics))
                if self.early_stop:
                    self.variable_dict[metrics]['wait'] += 1
                    if self.variable_dict[metrics]['wait'] > self.patient:
                        self.variable_dict[metrics]['stoppable'] = True

        else:
            raise ValueError('Unknown mode! The mode should be set to be "max" or "min"!')

    def run(self, session, current_value, current_epoch):
        if type(current_value) == float:
            current_value = [current_value]
        elif type(current_value) != list:
            raise ValueError('Unacceptable value type!')
        if current_value:
            self.variable_dict['stop_training'] = True
            for metric, value, m in zip(self.metrics, current_value, self.mode):
                self.save_decision(session, self.ckpt_path, metric, value, m, current_epoch)
                self.variable_dict['stop_training'] = self.variable_dict['stop_training'] and \
                                                      self.variable_dict[metric]['stoppable']