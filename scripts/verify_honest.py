"""Honest verification for the TRAINED system. Real pipeline calls only, no hardcoded values.

Validates:
- Trained MLP classifier loads from outputs/models/best_model.pkl
- Fire/smoke/no_fire test images produce real contract predictions
- Growth is always INSUFFICIENT_TEMPORAL_DATA for single images
- Severity/risk come from transparent disclosed rules on real probabilities
- No fake values: untrained components stay unavailable
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.demo_pipeline import DemoForestFirePipeline
from ai.lstm.lstm_temporal import LSTMWildfireRiskPredictor
from ai.efficientnet.efficientnet_classifier import EfficientNetClassifier
from ai.yolo.yolo_detector import YoloFireSmokeDetector

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(('PASS' if cond else 'FAIL') + ': ' + name + ' | ' + str(detail)[:180])

pipe = DemoForestFirePipeline()
print('MODEL_STATUS:', pipe.model_status)
check('image classifier trained', pipe.image_model is not None, pipe.model_status.get('image_classifier'))
check('efficientnet honest no fake', pipe.efficientnet.available is False, pipe.model_status.get('efficientnet'))
check('yolo honest no fake boxes', pipe.yolo.available is False, pipe.model_status.get('yolo'))
check('lstm honest no fake growth', pipe.temporal.loaded is False, pipe.model_status.get('temporal'))
check('efficientnet wrapper None w/o weights', EfficientNetClassifier(None).predict_image('nonexistent.jpg') is None, 'None')
check('yolo wrapper empty w/o ckpt', YoloFireSmokeDetector(None).detect('nonexistent.jpg') == [], '[]')
# New transparent severity rule on real evidence:
s_fire = DemoForestFirePipeline._severity('fire', None, {'fire': 0.9, 'smoke': 0.05, 'no_fire_smoke': 0.05}, backend_trained=True, confidence=0.9)
check('severity trained fire scored', s_fire[0] in {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'} and s_fire[1] is not None, s_fire)
s_un = DemoForestFirePipeline._severity('fire', None, {'fire': 0.9}, backend_trained=False)
check('severity untrained fire unavailable', s_un == ('Severity Unavailable', None), 'unavailable')
s_unc = DemoForestFirePipeline._severity('uncertain', None, {'fire': 0.4, 'smoke': 0.3, 'no_fire_smoke': 0.3}, backend_trained=True, confidence=0.4)
check('severity uncertain -> UNCERTAIN', s_unc == ('UNCERTAIN', None), s_unc)
s_no = DemoForestFirePipeline._severity('no_fire_smoke', None, {}, backend_trained=True)
check('severity trained nofire LOW 0.0', s_no == ('LOW', 0.0), 'LOW')
# Risk helper:
r1 = DemoForestFirePipeline._risk({'fire': 0.7, 'smoke': 0.2, 'no_fire_smoke': 0.1}, backend_trained=True)
check('risk trained fire HIGH 0.9', r1 == ('HIGH', 0.9), r1)
r0 = DemoForestFirePipeline._risk({'fire': 0.05, 'smoke': 0.05, 'no_fire_smoke': 0.9}, backend_trained=True)
check('risk trained nofire LOW 0.1', r0[1] == 0.1, r0)
r_un = DemoForestFirePipeline._risk({'fire': 0.9}, backend_trained=False)
check('risk untrained unavailable', r_un == ('Risk Unavailable', None), r_un)
lstm = LSTMWildfireRiskPredictor(None)
check('growth empty insufficient', lstm.predict_sequence([])['growth_probability'] is None, 'empty')
check('growth single insufficient', lstm.predict_sequence([0.9])['growth_probability'] is None, 'single')
check('growth seq no ckpt insufficient', lstm.predict_sequence([0.5, 0.6, 0.7])['growth_probability'] is None, lstm.predict_sequence([0.5, 0.6, 0.7]).get('reason'))
check('growth no frames insufficient', lstm.predict_frame_sequence([])['growth_probability'] is None, 'noframes')
# Real test images from the labeled dataset:
for cls in ['fire', 'smoke', 'no_fire']:
    imgs = sorted((ROOT / 'dataset' / 'test' / cls).glob('*'))
    if not imgs:
        check(f'{cls} test image exists', False, 'missing dataset')
        continue
    r = pipe.predict_file(str(imgs[0]))
    check(f'{cls}: prediction uppercase contract', r.get('prediction') in {'FIRE', 'SMOKE', 'NO_FIRE', 'UNCERTAIN'}, f"{imgs[0].name} => {r.get('prediction')}")
    check(f'{cls}: growth INSUFFICIENT_TEMPORAL_DATA', (r.get('growth') or {}).get('status') == 'INSUFFICIENT_TEMPORAL_DATA', (r.get('growth') or {}).get('status'))
    check(f'{cls}: severity has level', isinstance(r.get('severity'), dict) and r['severity'].get('level'), (r.get('severity') or {}).get('level'))
    check(f'{cls}: risk score present', isinstance(r.get('risk'), dict) and r['risk'].get('score') is not None, (r.get('risk') or {}).get('score'))
    check(f'{cls}: model trained', r.get('model_status') == 'trained', r.get('model_status'))
fails = [n for n, ok in results if not ok]
print('TOTAL ' + str(len(results)) + ' checks, ' + str(len(fails)) + ' failed: ' + str(fails))
sys.exit(1 if fails else 0)

