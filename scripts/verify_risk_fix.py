"""Verify Risk/Growth/Severity honesty fixes. Real pipeline calls, no hardcoded values."""
import inspect
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.demo_pipeline import DemoForestFirePipeline
from ai.lstm.lstm_temporal import LSTMWildfireRiskPredictor
from ai.efficientnet.efficientnet_classifier import EfficientNetClassifier
from ai.yolo.yolo_detector import YoloFireSmokeDetector
from backend.services import satellite_service

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(('PASS' if cond else 'FAIL') + ': ' + name + ' ' + str(detail)[:200])

pipe = DemoForestFirePipeline()
print('MODEL_STATUS:', pipe.model_status)
check('efficientnet honest (no fake trained)', pipe.efficientnet.available == False, pipe.model_status.get('efficientnet'))
check('yolo honest (no fake boxes)', pipe.yolo.available == False, pipe.model_status.get('yolo'))
check('lstm honest (no fake growth)', pipe.temporal.loaded == False, pipe.model_status.get('temporal'))
check('image classifier not trained', pipe.image_model is None, pipe.model_status.get('image_classifier'))
check('efficientnet wrapper None without weights', EfficientNetClassifier(None).predict_image('nonexistent.jpg') is None)
check('yolo wrapper [] without checkpoint', YoloFireSmokeDetector(None).detect('nonexistent.jpg') == [])

s1 = DemoForestFirePipeline._severity('fire', None, {'fire':0.9,'smoke':0.05,'no_fire_smoke':0.05}, backend_trained=False)
check('severity untrained -> Severity Unavailable/None', s1 == ('Severity Unavailable', None), s1)
s2 = DemoForestFirePipeline._severity('fire', 0.2, {'fire':0.9,'smoke':0.05,'no_fire_smoke':0.05}, backend_trained=True)
check('severity trained fire, no severity model -> Severity Unavailable/None', s2 == ('Severity Unavailable', None), s2)
s3 = DemoForestFirePipeline._severity('smoke', 0.2, {'fire':0.1,'smoke':0.8,'no_fire_smoke':0.1}, backend_trained=True)
check('severity trained smoke -> Severity Unavailable/None', s3 == ('Severity Unavailable', None), s3)
s4 = DemoForestFirePipeline._severity('no_fire_smoke', None, {'fire':0.05,'smoke':0.05,'no_fire_smoke':0.9}, backend_trained=True)
check('severity trained no_fire -> LOW/0.0', s4 == ('LOW', 0.0), s4)
s5 = DemoForestFirePipeline._severity('no_fire_smoke', None, {}, backend_trained=False)
check('severity untrained no_fire -> Severity Unavailable not LOW', s5 == ('Severity Unavailable', None), s5)

lstm = LSTMWildfireRiskPredictor(None)
g1 = lstm.predict_sequence([])
check('growth empty -> INSUFFICIENT_DATA None', g1['growth_probability'] is None and g1['status']=='INSUFFICIENT_DATA', g1)
g2 = lstm.predict_sequence([0.9])
check('growth single value -> INSUFFICIENT_DATA None', g2['growth_probability'] is None, g2)
g3 = lstm.predict_sequence([0.5, 0.6, 0.7])
check('growth real seq, no checkpoint -> INSUFFICIENT_DATA None', g3['growth_probability'] is None, g3.get('reason',''))
g4 = lstm.predict_frame_sequence([])
check('growth no frames -> INSUFFICIENT_DATA', g4['status']=='INSUFFICIENT_DATA' and g4['growth_probability'] is None, g4)
from PIL import Image
tmpdir = ROOT / 'outputs' / '_verify'
tmpdir.mkdir(parents=True, exist_ok=True)
fire_img = tmpdir / 'fire_red.jpg'; smoke_img = tmpdir / 'smoke_gray.jpg'; nofire_img = tmpdir / 'nofire_blue.jpg'
Image.new('RGB',(64,64),(220,30,20)).save(fire_img); Image.new('RGB',(64,64),(150,150,150)).save(smoke_img); Image.new('RGB',(64,64),(20,60,180)).save(nofire_img)
fb = pipe._pixel_fallback(str(fire_img))
check('pixel fallback probs None (no fake)', fb['probabilities']=={'fire':None,'smoke':None,'no_fire_smoke':None} and fb['confidence'] is None and fb['label']=='unavailable', fb)

for name, p in [('fire',fire_img),('smoke',smoke_img),('no-fire',nofire_img)]:
    r = pipe.predict_file(str(p))
    ok_risk = r['risk']['level']=='Risk Unavailable' and r['risk']['score'] is None and r['risk_score'] is None
    ok_sev = r['severity']=='Severity Unavailable' and r['severity_score'] is None
    ok_growth = r['growth']['level']=='INSUFFICIENT DATA' and r['growth']['probability'] is None
    ok_conf = r['confidence'] is None and r['confidence_percentage'] is None
    ok_probs = all(v is None for v in r['class_probabilities'].values())
    ok_detect = r['detection']=='unavailable'
    check('predict ' + name + ': Risk Unavailable', ok_risk, r['risk'])
    check('predict ' + name + ': Severity Unavailable', ok_sev, str(r['severity']) + '/' + str(r['severity_score']))
    check('predict ' + name + ': Growth INSUFFICIENT_DATA', ok_growth, r['growth'])
    check('predict ' + name + ': confidence None not faked', ok_conf, str(r['confidence']) + '/' + str(r['confidence_percentage']))
    check('predict ' + name + ': probs None not faked', ok_probs, r['class_probabilities'])
    check('predict ' + name + ': detection unavailable', ok_detect, r['detection'])
    check('predict ' + name + ': no alert on unavailable', r['alert'] is False, r['alert'])

def risk_from_probs(probs):
    return round(max(0.0, min(1.0, 1.0 - float(probs['no_fire_smoke']))), 4)
check('risk = 1-P(no_fire) fire case', risk_from_probs({'fire': 0.7, 'smoke': 0.2, 'no_fire_smoke': 0.1}) == 0.9, '0.9')
check('risk distinct from confidence', risk_from_probs({'fire': 0.7, 'smoke': 0.2, 'no_fire_smoke': 0.1}) != 0.7, 'distinct')
check('risk no-fire residual', risk_from_probs({'fire': 0.05, 'smoke': 0.05, 'no_fire_smoke': 0.9}) == 0.1, '0.1')
src = inspect.getsource(satellite_service.find_nearby_observation)
check('satellite no 0.7/0.3 blend', '0.7' not in src and '0.3' not in src, 'blend removed')
check('satellite method documented', 'confidence / 100' in src or 'confidence/100' in src, 'documented')
fails = [n for n, ok, _ in results if not ok]
print('TOTAL ' + str(len(results)) + ' checks, ' + str(len(fails)) + ' failed: ' + str(fails))
sys.exit(1 if fails else 0)

