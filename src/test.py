import argparse
import os

import data_loader.data_loaders as module_data
import torch
import numpy as np

import model.metric as module_metric
import model.model as module_arch
import model as module_loss
from train import get_instance
from utils.util import tokenizer
from utils.util import seq_mask


def build_data(data, device):
    # Note: the size of sequential data iter is [batch_seq_len, batch_size]
    input = {
        'question': data.question[0].transpose(0, 1),
        'question_len': data.question[1],
        'question_mask': seq_mask(data.question[1], device=device),
        'right_answer': data.answer[0].transpose(0, 1),
        'right_answer_len': data.answer[1],
        'right_answer_mask': seq_mask(data.answer[1], device=device),
        'wrong_answer': None,
        'wrong_answer_len': None,
        'wrong_answer_mask': None,
        'predict': True,
    }
    return input


def eval_metrics(scores, data_loader, metrics):
        metrics_result = np.zeros(len(metrics))
        question_count = 0
        last_question = None
        right_answer_scores = []
        wrong_answer_scores = []
        assert len(scores) == len(data_loader.test)
        for idx, line in enumerate(data_loader.test):
            question, answer, label, score = line.question, line.answer, line.label, scores[idx]
            print('%s\t%s\t%s\t%f\t' % (' '.join(question), ' '.join(answer), label, score))
            if question != last_question and last_question != None:
                # get MAP and MRR
                if len(right_answer_scores) != 0 and len(wrong_answer_scores) != 0:
                    for i, metric in enumerate(metrics):
                        mrr = metric(right_answer_scores, wrong_answer_scores)
                        print(right_answer_scores, wrong_answer_scores)
                        print('MRR:%f'%(mrr))
                        metrics_result[i] += mrr
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
            for i, metric in enumerate(metrics):
                metrics_result[i] += metric(right_answer_scores, wrong_answer_scores)
            question_count += 1
            # get the final MRR and MAP
            metrics_result = np.array([i/question_count for i in metrics_result])

        return metrics_result


def main(config, resume):
    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config, tokenizer, )

    # build model architecture
    model = get_instance(module_arch, 'arch', config, data_loader.train_vocab.vectors)
    model.summary()

    # get function handles of loss and metrics
    # loss = getattr(module_loss, config['loss']['type'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader.test_iter):
            input = build_data(data, device)
            output =model(input)
            # loss = self.loss(output[0], output[1], self.config['loss']['t0'], self.device)

            # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            # self.writer.add_scalar('loss', loss.item())

            # record the score to eval
            scores.extend(output[0].tolist())
            # evaluate
        total_val_metrics = eval_metrics(scores, data_loader, metrics)
    print('MAP:', total_val_metrics)
    # print('dev loss:', total_val_loss / len(self.data_loader.eval_iter))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)
