#!/usr/bin/env python3
"""Plot torch.cuda.Event timing breakdown per action chunk.

Reads one or more jsonlines files produced by `CudaTimerCollection.flush_to_file`
and produces one stacked bar chart per `action_chunk` (one PNG each).

Usage:
    python scripts/plot_torch_cuda_event_breakdown.py \
        --input outputs/action_conditioned/torch_cuda_event/torch_cuda_event/0.jsonl \
        --outdir outputs/action_conditioned/figs --granularity 0 --topk 20

Granularity (int): controls how module names are aggregated.
 - The script looks for the marker `_checkpoint_wrapped_module` in module names
   (e.g. `blocks.0._checkpoint_wrapped_module.self_attn.q_proj`).
 - granularity=0 -> group to `blocks.0._checkpoint_wrapped_module.<first>` (e.g. `...self_attn`)
 - granularity=1 -> group to `blocks.0._checkpoint_wrapped_module.<first>.<second>` (e.g. `...self_attn.q_proj`)
 - If `_checkpoint_wrapped_module` not found, fallback to simple dot-splitting.

Output: one PNG per action_chunk named `action_chunk_{id}.png` in `--outdir`.
"""
import argparse
import glob
import json
import os
from collections import defaultdict, Counter
from typing import List, Dict

import matplotlib.pyplot as plt

# ---------- Top-level configurable defaults ----------
# Set these values to control default behavior without CLI args.
# `run_id` will be interpolated into the path templates below.
DEFAULT_RUN_ID = "torch_cuda_event"
# DEFAULT_RUN_ID = "torch_cuda_event_full_denoise"
DEFAULT_INPUT_TEMPLATE = "outputs/action_conditioned/{run_id}/torch_cuda_event/*.jsonl"
DEFAULT_OUTDIR_TEMPLATE = "outputs/action_conditioned/{run_id}/figs"
DEFAULT_GRANULARITY = 0
DEFAULT_TOPK = 15


def load_records(paths: List[str]):
    records = []
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except Exception:
                    # skip malformed
                    continue
    return records


def group_module_name(name: str, granularity: int) -> str:
    parts = name.split('.') if name is not None else []
    if not parts:
        return name or 'unknown'
    # find marker
    try:
        idx = parts.index('_checkpoint_wrapped_module')
        base = '.'.join(parts[: idx + 1])
        rest = parts[idx + 1 :]
        take = rest[: max(1, granularity + 1) ] if rest else []
        if take:
            return base + '.' + '.'.join(take)
        else:
            return base
    except ValueError:
        # fallback: take first (granularity+2) parts
        take_n = min(len(parts), granularity + 2)
        return '.'.join(parts[:take_n])


def aggregate_for_chunk(records: List[Dict], granularity: int, topk: int = 20, verbose: bool = False, no_aggregate: bool = False):
    # records filtered for single action_chunk
    # map: (denoise_step, block_index) -> group -> total_ms
    def extract_block_index(layer_name: str) -> int:
        if not layer_name:
            return -1
        parts = layer_name.split('.')
        for i, p in enumerate(parts):
            if p == 'blocks' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except Exception:
                    return -1
        return -1

    agg = defaultdict(lambda: defaultdict(float))
    group_totals = Counter()
    keys_set = set()
    agg_counts = defaultdict(int)
    key_records = defaultdict(list)
    total_json_ms = 0.0
    total_records = 0
    skipped_marker_single = 0
    for r in records:
        step = r.get('denoise_step')
        layer = r.get('layer_name', '')
        # Filtering rules based on granularity requested by user:
        # - granularity == 1: skip records where `_checkpoint_wrapped_module` is
        #   followed by zero or only one trailing segment (we want deeper names).
        # - granularity == 0: only keep records where `_checkpoint_wrapped_module`
        #   is followed by exactly one trailing segment (e.g. `...self_attn`);
        #   skip records with more trailing segments and also skip records
        #   without the marker.
        if layer:
            parts_check = layer.split('.')
            try:
                idx_marker = parts_check.index('_checkpoint_wrapped_module')
                tail_len = len(parts_check) - idx_marker - 1
                if granularity == 1:
                    # skip entries where there's zero or only one field after marker
                    if tail_len <= 1:
                        skipped_marker_single += 1
                        continue
                elif granularity == 0:
                    # only keep entries where there's exactly one field after marker
                    if tail_len != 1:
                        # not the simple `..._checkpoint_wrapped_module.<single>` form -> skip
                        skipped_marker_single += 1
                        continue
            except ValueError:
                # marker not present
                if granularity == 1:
                    # nothing special to skip when granularity==1
                    pass
                else:
                    # granularity==0: user asked to focus on marker-based simple names
                    skipped_marker_single += 1
                    continue
        else:
            # no layer name
            if granularity == 0:
                skipped_marker_single += 1
                continue
        block_idx = extract_block_index(layer)
        key = (step, block_idx)
        group = group_module_name(layer, granularity)
        display_group = group
        # If the user requested no aggregation, make each JSON record a
        # distinct display label so we do not sum multiple records into a
        # single group. We append the global record index to guarantee
        # uniqueness.
        if no_aggregate:
            display_group = f"{display_group}__rec{total_records}"

        if isinstance(display_group, str):
            # 1) Remove the leading marker and everything before it if present
            if "_checkpoint_wrapped_module." in display_group:
                # e.g. "blocks.0._checkpoint_wrapped_module.self_attn.q_proj"
                # -> "self_attn.q_proj"
                display_group = display_group.split("_checkpoint_wrapped_module.", 1)[1]
            # 2) If the marker was not present but the name has a leading
            #    "blocks.<idx>." prefix, strip that prefix
            elif display_group.startswith("blocks."):
                parts = display_group.split(".", 2)
                if len(parts) > 2:
                    # "blocks.0.something" -> "something"
                    display_group = parts[2]

            # 3) For safety: drop leading underscores so legend entries are not
            #    harder to read or accidentally ignored
            if display_group.startswith("_"):
                display_group = display_group.lstrip("_")

        ms = float(r.get('elapsed_ms', 0.0) or 0.0)
        total_json_ms += ms
        total_records += 1
        agg[key][display_group] += ms
        agg_counts[key] += 1
        key_records[key].append(r)
        group_totals[display_group] += ms
        keys_set.add(key)

    # build matrix: rows are (step, block) sorted by step desc, block asc
    keys = sorted(list(keys_set), key=lambda x: (-(x[0] if x[0] is not None else -1), x[1]))

    # If granularity==1, only plot block index 5 for each denoise step.
    # Filter the keys to only include entries with block_idx == 5.
    if granularity == 1:
        # keep only block index 5 (user requested) and report how many keys remain
        orig_key_count = len(keys)
        keys = [k for k in keys if k[1] == 5]
        filtered_out_keys = orig_key_count - len(keys)
        if verbose:
            print(f"[agg] granularity=1: kept {len(keys)} keys, filtered out {filtered_out_keys} keys (non-block-5)")

    # choose top-k groups by total ms. When we've filtered keys (granularity==1)
    # compute totals from the filtered keys only so the top-k reflects the
    # selected block's profile rather than the whole model.
    if granularity == 1:
        filtered_totals = Counter()
        for key in keys:
            for g, v in agg.get(key, {}).items():
                filtered_totals[g] += v
        top_groups = [g for g, _ in filtered_totals.most_common(topk)]
    else:
        # When no_aggregate is requested, include all observed groups so
        # segments are not merged into 'other'. Otherwise pick top-k by
        # total ms.
        if no_aggregate:
            top_groups = list(group_totals.keys())
        else:
            top_groups = [g for g, _ in group_totals.most_common(topk)]
    table = []
    for key in keys:
        row = {}
        total_other = 0.0
        for g, v in agg.get(key, {}).items():
            if g in top_groups:
                row[g] = v
            else:
                total_other += v
        row['other'] = total_other
        table.append((key, row))

    groups = top_groups + ['other']
    if verbose:
        # compute totals included in the final table (after key filtering)
        included_ms = 0.0
        included_records = 0
        for key, row in table:
            included_ms += sum(row.values())
            included_records += agg_counts.get(key, 0)
        print(f"[agg] total JSON records: {total_records}, total JSON ms: {total_json_ms:.6f}")
        print(f"[agg] included records in table: {included_records}, included ms (plotted): {included_ms:.6f}")
        print(f"[agg] skipped_marker_single: {skipped_marker_single}")
        # Print a detailed breakdown for the first plotted bar (if any)
        if table:
            first_key, first_row = table[0]
            print(f"[first_bar] key=(denoise_step,block)={first_key}")
            # sort groups by descending ms for readability
            items = sorted(first_row.items(), key=lambda x: -x[1])
            for g, v in items:
                print(f"  {g}: {v:.6f} ms")
            recs = key_records.get(first_key, [])
            print(f"[first_bar] contributing JSON records: {len(recs)} (showing up to 10)")
            for i, rec in enumerate(recs[:100]):
                print(f"  #{i+1}: denoise_step={rec.get('denoise_step')}, layer_name={rec.get('layer_name')}, elapsed_ms={rec.get('elapsed_ms')}")
    return keys, groups, table


def plot_chunk(steps, groups, table, outpath, title=None):
    # build stacked bars where each entry in `steps` is (denoise_step, block_idx)
    x = list(range(len(steps)))
    # prepare bottoms
    bottoms = [0.0] * len(steps)
    width = max(6, len(steps) * 0.5)
    height = 6
    fig, ax = plt.subplots(figsize=(width, height))
    # add extra width to accommodate the vertical legend on the right
    extra_legend_width = min(max(1.5, len(groups) * 0.12), 12)
    fig.set_size_inches(width + extra_legend_width, height)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(groups))]
    for i, g in enumerate(groups):
        vals = [table[j][1].get(g, 0.0) for j in range(len(steps))]
        # print(vals)
        ax.bar(x, vals, bottom=bottoms, label=g, color=colors[i])
        bottoms = [bottoms[j] + vals[j] for j in range(len(steps))]

    ax.set_xticks(x)
    # label ticks as "step:block"
    ax.set_xticklabels([f"{s[0]}:{s[1]}" for s in steps], rotation=45, ha='right')
    ax.set_xlabel('denoise_step')
    ax.set_ylabel('elapsed_ms (sum)')
    if title:
        ax.set_title(title)
    # place vertical legend on the right, centered; one column so entries are stacked
    legend_fontsize = 10
    # ax.legend(loc='upper left', fontsize=legend_fontsize)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=legend_fontsize)
    fig.tight_layout()
    # use bbox_inches='tight' so the legend outside the axes is included in the saved image
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', nargs='+', required=False, help='Input jsonl file(s) or glob (optional if --run-id is provided)')
    p.add_argument('--outdir', '-o', required=False, default=None, help='Output directory for figs (optional if --run-id is provided)')
    p.add_argument('--run-id', '-r', required=False, default=DEFAULT_RUN_ID, help='Run id used to build input/outdir templates')
    p.add_argument('--granularity', '-g', type=int, default=DEFAULT_GRANULARITY, help='Grouping granularity (0 or 1)')
    p.add_argument('--topk', type=int, default=DEFAULT_TOPK, help='Top-K groups to show; rest -> other')
    p.add_argument('--verbose', action='store_true', help='Print diagnostic totals and counts')
    p.add_argument('--no-aggregate', action='store_true', help='Do not aggregate multiple JSON records into the same group; plot each record as its own segment')
    args = p.parse_args()

    # Resolve input files and output dir.
    input_paths = []
    if args.run_id is not None:
        # expand run-based template
        template = DEFAULT_INPUT_TEMPLATE
        glob_path = template.format(run_id=args.run_id)
        input_paths = sorted(glob.glob(glob_path))
        # if user explicitly provided --input, append those as well
        if args.input:
            input_paths = sorted(set(input_paths + args.input))
        outdir = args.outdir or DEFAULT_OUTDIR_TEMPLATE.format(run_id=args.run_id)
    else:
        input_paths = args.input or []
        outdir = args.outdir

    if not input_paths:
        print('No input files found. Provide --input or --run-id that matches files.')
        return

    os.makedirs(outdir, exist_ok=True)
    recs = load_records(input_paths)
    if not recs:
        print('No records read from inputs')
        return

    # group by action_chunk
    by_chunk = defaultdict(list)
    for r in recs:
        chunk = r.get('action_chunk')
        if chunk is None:
            chunk = -1
        by_chunk[chunk].append(r)

    for chunk, chunk_recs in sorted(by_chunk.items()):
        steps, groups, table = aggregate_for_chunk(
            chunk_recs, args.granularity, topk=args.topk, verbose=args.verbose, no_aggregate=args.no_aggregate
        )
        if not steps:
            print(f'No steps for chunk {chunk}, skipping')
            continue
        outpath = os.path.join(outdir, f'gran_{args.granularity}_action_chunk_{chunk}.png')
        title = f'Action chunk {chunk} (groups={len(groups)})'
        plot_chunk(steps, groups, table, outpath, title=title)
        print(f'Wrote {outpath}')


if __name__ == '__main__':
    main()
