import os

import argparse

from QDeepNetwork import QNAgent, TrainingSupervisor
from DinoGameRunner import FreezingGameRunner

if __name__ == '__main__':
    working_dir = os.path.dirname(os.path.realpath(__file__))
    t_rex_runner_url = 'file://' + working_dir + '/t-rex-runner/index.html'

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_weights', '-o', action='store')
    parser.add_argument('--training_sessions', '-n', type=int, action='store')
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--input_weights', '-i', action='store', required=False)
    parser.add_argument('--replay_batch_size', '-r', action='store', type=int, default=64)
    parser.add_argument('--replay_error_batch_size', action='store', type=int, default=8)
    parser.add_argument('--pure_exploration_turns', '-e', action='store', type=int, default=100)
    parser.add_argument('--game_path', action='store', default=t_rex_runner_url)

    args = parser.parse_args()

    qnagent = QNAgent((50, 135, 1))

    if args.input_weights is not None:
        qnagent.load(args.input_weights)
        print('=== Initial weights loaded from', args.input_weights)

    game_runner = FreezingGameRunner(args.game_path, args.headless)
    training_supervisor = TrainingSupervisor(qnagent, game_runner)

    print('=== Begin', args.training_sessions, 'sessions of training')

    for n in range(args.training_sessions):
        print('(', n + 1, '/', args.training_sessions, ')', sep='')
        score = training_supervisor.train(args.replay_batch_size, args.replay_error_batch_size, force_explore=(n < args.pure_exploration_turns))
        print('\t\t\t\t\tScore', score)

    print('=== Training finished')
    print('Top 10 high scores:', training_supervisor.get_high_scores())
    print('Average score:', training_supervisor.get_average_score())
    print('Last 50 average:', training_supervisor.get_average_score(50))
    print('Last 100 average', training_supervisor.get_average_score(100))
    print('Last 200 average:', training_supervisor.get_average_score(200))

    print('=== Saving weights to', args.output_weights)
    qnagent.save(args.output_weights)
