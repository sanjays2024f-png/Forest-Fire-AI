import { MapContainer, Marker, Popup, TileLayer } from 'react-leaflet';
import L from 'leaflet';

const fireIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const smokeIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-orange.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

export default function AlertMap({ result }) {
  const hasLocation = result?.latitude !== null && result?.latitude !== undefined
    && result?.longitude !== null && result?.longitude !== undefined;
  const center = hasLocation ? [result.latitude, result.longitude] : [20, 0];
  return (
    <div className="map-wrapper card">
      <div className="section-header">
        <h3>Alert Map</h3>
      </div>
      {hasLocation ? (
      <MapContainer center={center} zoom={6} scrollWheelZoom className="leaflet-map">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Marker position={center} icon={result.detection.type.includes('smoke') ? smokeIcon : fireIcon}>
          <Popup>{result.alert.level}</Popup>
        </Marker>
      </MapContainer>
      ) : <div className="location-unavailable">Location unavailable</div>}
    </div>
  );
}
