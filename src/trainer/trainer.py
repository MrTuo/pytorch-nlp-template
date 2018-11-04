import numpy as np
import torch

from base import BaseTrainer
from utils.util import seq_mask

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.do_validation = self.data_loader.eval_iter is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.train_batch_size))

    def _eval_metrics(self, scores):
        metrics_result = np.zeros(len(self.metrics))
        question_count = 0
        last_question = None
        right_answer_scores = []
        wrong_answer_scores = []
        assert len(scores) == len(self.data_loader.eval)
        for idx, line in enumerate(self.data_loader.eval):
            question , answer, label, score = line.question, line.answer, line.label, scores[idx]
            if question != last_question and last_question != None:
                # get MAP and MRR
                if len(right_answer_scores) != 0 and len(wrong_answer_scores) != 0:
                    for i, metric in enumerate(self.metrics):
                        metrics_result[i] += metric(right_answer_scores, wrong_answer_scores)
                    question_count += 1
                # reset scores lists
                right_answer_scores = []
                wrong_answer_scores = []

            # save scores
            if label == '1':
                right_answer_scores.append(score)
            else:
                wrong_answer_scores.append(score)
            last_question = question

        if last_question != None:
            # get MRR and MAP of the last question
            for i, metric in enumerate(self.metrics):
                metrics_result[i] += metric(right_answer_scores, wrong_answer_scores)
            question_count += 1
            # get the final MRR and MAP
            metrics_result = np.array([i/question_count for i in metrics_result])

        return metrics_result

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        total_loss = 0

        self.data_loader.train_iter.device = self.device
        for batch_idx, data in enumerate(self.data_loader.train_iter):
            input = self.build_data(data)

            self.optimizer.zero_grad()
            output = self.model(input)
            # print(output[0])
            loss = self.loss(output[0], output[1], self.config['loss']['t0'], self.device)
            loss.backward()
            self.optimizer.step()

            # self.writer.set_step((epoch - 1) * len(self.data_loader.train_iter) + batch_idx)
            # self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            # print('Loss:', loss.item())

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.train_batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader.train_iter),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0

        scores = []
        with torch.no_grad():
            self.data_loader.train_iter.device = self.device
            for batch_idx, data in enumerate(self.data_loader.eval_iter):
                input = self.build_data(data, eval=True)
                output = self.model(input)
                # loss = self.loss(output[0], output[1], self.config['loss']['t0'], self.device)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # self.writer.add_scalar('loss', loss.item())

                # record the score to eval
                scores.extend(output[0].tolist())
        # evaluate
        total_val_metrics = self._eval_metrics(scores)
        print('MAP & MRR:', total_val_metrics)
        # print('dev loss:', total_val_loss / len(self.data_loader.eval_iter))
        return {
            # 'val_loss': total_val_loss / len(self.data_loader.eval_iter),
            'val_metrics': total_val_metrics,
            'val_map': total_val_metrics[0],
        }

    def build_data(self, data, eval=False):
        # Note: the size of sequential data iter is [batch_seq_len, batch_size]
        input = {
            'question': data.question[0].transpose(0, 1),
            'question_len': data.question[1],
            'question_mask': seq_mask(data.question[1], device=self.device),
            'right_answer': data.right_answer[0].transpose(0, 1) if not eval else data.answer[0].transpose(0, 1),
            'right_answer_len': data.right_answer[1] if not eval else data.answer[1],
            'right_answer_mask': seq_mask(data.right_answer[1], device=self.device) if not eval else seq_mask(data.answer[1], device=self.device),
            'wrong_answer': data.wrong_answer[0].transpose(0, 1) if not eval else None,
            'wrong_answer_len': data.wrong_answer[1] if not eval else None,
            'wrong_answer_mask': seq_mask(data.wrong_answer[1], device=self.device) if not eval else None,
            'predict': False if not eval else True,
        }

        return input

    @staticmethod
    def get_mrr(right_answer_scores, wrong_answer_scores):
        all_scores_ranked = (right_answer_scores + wrong_answer_scores).sort(reverse=True)
        mrr = 0
        for right_score in right_answer_scores:
            rank = all_scores_ranked.index(right_score) + 1
            mrr += 1.0 / rank
        return mrr

    @staticmethod
    def get_map(right_answer_scores, wrong_answer_scores):
        all_scores_ranked = (right_answer_scores + wrong_answer_scores).sort(reverse=True)
        right_answer_scores_ranked = sorted(right_answer_scores.copy(), reverse=True)
        map = 0
        for idx, right_score in enumerate(right_answer_scores_ranked):
            rank = all_scores_ranked.index(right_score) + 1
            map += (idx+1) / rank
        return map
