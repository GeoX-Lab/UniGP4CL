"""Public UniGP class-incremental classification entry point."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='UniGP: open and historical geometry protection')
    parser.add_argument('--dataset', required=True, choices=('aid', 'nwpu', 'ucm'))
    parser.add_argument('--data-root', required=True, help='Dataset root (NWPU: directory containing train/train and test/test)')
    parser.add_argument('--ucm-classes-per-task', type=int, choices=(3, 7), default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--samples-per-class', type=int, default=20)
    parser.add_argument('--ogp-weight', type=float, default=0.5)
    parser.add_argument('--hgp-weight', type=float, default=0.5)
    parser.add_argument('--hgp-temperature', type=float, default=2.0)
    parser.add_argument('--max-stages', type=int, default=None)
    args = parser.parse_args()
    protocol, tasks, classes = {'aid': ('aid30_6x5', 6, 5), 'nwpu': ('nwpu45', 5, 9), 'ucm': (f'ucm21_{21 // args.ucm_classes_per_task}x{args.ucm_classes_per_task}', 21 // args.ucm_classes_per_task, args.ucm_classes_per_task)}[args.dataset]
    root = Path(args.data_root).expanduser().resolve()
    if args.dataset == 'ucm' and (root / 'Images').is_dir():
        root = root / 'Images'
    if not root.is_dir():
        parser.error(f'data root does not exist: {root}')
    tag = args.tag or f'{protocol}_unigp_seed{args.seed}'
    command = ['unigp.train', '--tag', tag, '--protocol', protocol, '--n-tasks', str(tasks), '--classes-per-task', str(classes), '--data-root', str(root), '--seed', str(args.seed), '--class-order-seed', str(args.seed), '--split-seed', str(args.split_seed), '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--lr', str(args.lr), '--train-scope', 'vision', '--classification-scope', 'current', '--ogp-weight', str(args.ogp_weight), '--hgp-weight', str(args.hgp_weight), '--hgp-temperature', str(args.hgp_temperature), '--samples-per-class', str(args.samples_per_class)]
    if args.max_stages is not None:
        command += ['--max-stages', str(args.max_stages)]
    sys.argv = command
    from unigp.train import train_unigp
    train_unigp()


if __name__ == '__main__':
    main()
