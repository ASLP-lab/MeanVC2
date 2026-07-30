"""
Generate training filelist.

Usage:
  python scripts/create_filelist.py \
    --bn-dir /path/to/bn \
    --mel-dir /path/to/mel \
    --xvector-dir /path/to/xvector \
    --output train.list

Output format (one line per utterance):
  utt_id|/path/to/bn/utt_id.npy|/path/to/mel/utt_id.npy|/path/to/xvector/utt_id.npy
"""
import argparse, os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bn-dir', required=True)
    parser.add_argument('--mel-dir', required=True)
    parser.add_argument('--xvector-dir', required=True)
    parser.add_argument('--output', default='train.list')
    args = parser.parse_args()

    # Collect all utt_ids (intersection of npy files across all three directories)
    def get_ids(d):
        return {f.replace('.npy', '') for f in os.listdir(d) if f.endswith('.npy')}

    bn_ids = get_ids(args.bn_dir)
    mel_ids = get_ids(args.mel_dir)
    xv_ids = get_ids(args.xvector_dir)

    common = sorted(bn_ids & mel_ids & xv_ids)
    print(f"BN: {len(bn_ids)}, Mel: {len(mel_ids)}, XV: {len(xv_ids)} → Common: {len(common)}")

    if not common:
        print("ERROR: No common IDs found!")
        return

    lines = []
    for uid in common:
        bn_path = os.path.join(args.bn_dir, f"{uid}.npy")
        mel_path = os.path.join(args.mel_dir, f"{uid}.npy")
        xv_path = os.path.join(args.xvector_dir, f"{uid}.npy")
        lines.append(f"{uid}|{bn_path}|{mel_path}|{xv_path}")

    with open(args.output, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Done: {len(lines)} lines → {args.output}")


if __name__ == "__main__":
    main()
