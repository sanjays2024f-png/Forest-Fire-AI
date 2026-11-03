const textOrNull = (value) => (value === null || value === undefined || value === '' ? null : String(value));
const numberOrNull = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);

const PRED_MAP = { FIRE: 'fire', SMOKE: 'smoke', NO_FIRE: 'no_fire_smoke', UNCERTAIN: 'uncertain', 'MODEL NOT TRAINED': 'unavailable' };

export function normalizeAnalysisResult(response) {
  const source = response || {};
  const predUpper = textOrNull(source.prediction);
  let detectionType = predUpper && PRED_MAP[predUpper] ? PRED_MAP[predUpper] : null;
  if (!detectionType) {
    const detectionSource = source.detection_result && typeof source.detection_result === 'object'
      ? source.detection_result
      : source.detection && typeof source.detection === 'object'
      ? source.detection
      : {};
    detectionType = textOrNull(
      detectionSource.type || (typeof source.detection === 'string' ? source.detection : null) || 'unavailable',
    );
  }
  const confidence = numberOrNull(source.confidence);
  const severitySource = source.severity_result || (source.severity && typeof source.severity === 'object' ? source.severity : {});
  const growthSource = source.growth && typeof source.growth === 'object' ? source.growth : {};
  const satelliteSource = source.satellite && typeof source.satellite === 'object' ? source.satellite : {};
  const probsSource = source.probabilities && typeof source.probabilities === 'object'
    ? source.probabilities
    : source.class_probabilities && typeof source.class_probabilities === 'object'
    ? { fire: source.class_probabilities.fire, smoke: source.class_probabilities.smoke, no_fire: source.class_probabilities.no_fire_smoke }
    : source.efficientnet && typeof source.efficientnet.probabilities === 'object'
    ? source.efficientnet.probabilities
    : {};
  const riskSource = source.risk && typeof source.risk === 'object' ? source.risk : {};
  const probs = {
    fire: numberOrNull(probsSource.fire),
    smoke: numberOrNull(probsSource.smoke),
    no_fire: numberOrNull(probsSource.no_fire_smoke ?? probsSource.no_fire),
  };
  return {
    raw: source,
    timestamp: textOrNull(source.timestamp),
    latitude: numberOrNull(source.latitude),
    longitude: numberOrNull(source.longitude),
    locationStatus: textOrNull(source.location_status) || 'Location unavailable',
    prediction: predUpper,
    detection: { type: detectionType, confidence },
    confidence,
    confidencePercentage: confidence === null ? null : confidence * 100,
    probabilities: probs,
    severity: {
      level: textOrNull(severitySource.level || source.severity_level || (typeof source.severity === 'string' ? source.severity : null)),
      score: numberOrNull(severitySource.score ?? source.severity_score),
      method: textOrNull(severitySource.method),
    },
    growth: {
      probability: numberOrNull(growthSource.probability ?? source.growth_probability),
      level: textOrNull(growthSource.level || growthSource.status || source.growth_label) || 'INSUFFICIENT_TEMPORAL_DATA',
      status: textOrNull(growthSource.status) || 'INSUFFICIENT_TEMPORAL_DATA',
      message: textOrNull(growthSource.message || growthSource.reason),
    },
    satellite: {
      risk: textOrNull(satelliteSource.risk),
      riskScore: numberOrNull(satelliteSource.risk_score),
      available: satelliteSource.available === true,
      status: textOrNull(satelliteSource.status) || 'Satellite data unavailable',
      source: textOrNull(satelliteSource.source) || 'MODIS 2024',
    },
    affectedArea: {
      value: numberOrNull(typeof source.affected_area === 'number' ? source.affected_area : null),
      unit: textOrNull(source.area_unit) || 'unavailable',
    },
    overallRisk: {
      score: numberOrNull(riskSource.score ?? source.risk_score),
      level: textOrNull(riskSource.level),
      method: textOrNull(riskSource.method),
    },
    boxes: Array.isArray(source.bounding_boxes) ? source.bounding_boxes : [],
    alert: { level: textOrNull(source.alert_status) || 'NO ALERT' },
    modelUsed: textOrNull(source.model_used),
    modelStatus: textOrNull(source.model_status),
    evidence: Array.isArray(source.evidence) ? source.evidence.map(String) : [],
  };
}
