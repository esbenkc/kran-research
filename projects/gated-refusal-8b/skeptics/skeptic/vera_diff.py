"""Vera: field-by-field diff of my rebuild (check_s012.json, from vera_check.py) against the build's analysis.json
(judge block). Usage: python vera_diff.py <analysis.json> <check_s012.json>"""
import json, sys

a = json.load(open(sys.argv[1]))['judge']
v = json.load(open(sys.argv[2]))
rows, bad = 0, []


def eq(name, x, y, tol=6e-5):
    global rows
    rows += 1
    ok = abs(float(x) - float(y)) <= tol if isinstance(x, float) or isinstance(y, float) else x == y
    if not ok:
        bad.append((name, x, y))


arm_map = {'S': 'strip', 'G': 'gated', 'U': 'ungated', 'O': 'oracle', 'RP': 'random_prompt_gate', 'C': 'canned'}
for srow in a['seeds']:
    s = srow['seed']
    o = v['seeds'][str(s)]
    eq(f's{s}.B', srow['B'], v['B'])
    eq(f's{s}.L_read', srow['L_read'], o['L_read'])
    eq(f's{s}.cut', srow['cut'], o['cut'])
    eq(f's{s}.push', f"L{srow['L_push']}_g{srow['g']}", o['push_pick'])
    for k, val in srow['arms'].items():
        if k in arm_map:
            mine = o[arm_map[k]]
            for f, idx in (('harmful_refused_n', 0), ('restore', 1), ('new_harmless_n', 2)):
                if f in val:
                    eq(f's{s}.{k}.{f}', val[f], mine[idx])
        if k in ('RG0', 'RG1'):
            mine = o['random_gates'][int(k[-1])]
            for f, idx in (('harmful_refused_n', 0), ('restore', 1), ('new_harmless_n', 2)):
                if f in val:
                    eq(f's{s}.{k}.{f}', val[f], mine[idx])
    eq(f's{s}.pred', srow['dev']['pred_s'], o['pred'], 6e-5)
    eq(f's{s}.recall_dev', srow['dev']['recall_dev'], o['pred_parts']['recall_dev'])
    eq(f's{s}.conv_dev', srow['dev']['conv_dev'], o['pred_parts']['conv_dev'])
    eq(f's{s}.strip_dev', srow['dev']['strip_dev'], o['pred_parts']['strip_dev'])
    eq(f's{s}.harmless_flagged', srow['harmless_flagged_test'], o['test_flag_harmless'])
    eq(f's{s}.strip_gate', srow['strip_gate_ok'], o['strip_gate_pass'])
sm, vs = a['summary'], v['summary']
eq('G.restore_mean', sm['G']['restore_mean'], vs['mean_gated_restore'][0])
eq('G.restore_t95.lo', sm['G']['restore_seed_t95'][0], vs['mean_gated_restore'][1])
eq('G.restore_t95.hi', sm['G']['restore_seed_t95'][1], vs['mean_gated_restore'][2])
eq('G.new_mean', sm['G']['new_harmless_mean'], vs['mean_gated_new_harmless'][0], 6e-4)
eq('G.new_t95.lo', sm['G']['new_harmless_seed_t95'][0], vs['mean_gated_new_harmless'][1], 6e-4)
eq('G.new_t95.hi', sm['G']['new_harmless_seed_t95'][1], vs['mean_gated_new_harmless'][2], 6e-4)
eq('RG_all', sm['RG_all']['restore_mean'], vs['mean_random_gate_restore'])
eq('status', a['status']['status'], vs['status'])
eq('conds', [a['status'][k] for k in ('c1_restore_ge_0.70', 'c2_new_harmless_le_5', 'c3_random_gate_lt_0.30')],
   vs['conds'])
eq('pred_mean', a['status']['secondary']['pred_mean'], vs['mean_pred'])
eq('secondary_holds', a['status']['secondary']['holds'], vs['secondary_holds'])
eq('prediction', a['status']['prediction'], 'failed' if vs['scout_prediction_held'] is False else vs['scout_prediction_held'])
print(f'{rows} fields compared, {len(bad)} mismatches')
for b in bad:
    print('MISMATCH', b)
