export default function DashboardCard({ title, value, subtitle, accent = 'blue' }) {
  return (
    <div className={`card metric-card ${accent}`}>
      <div className="metric-label">{title}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-subtitle">{subtitle}</div>
    </div>
  );
}
