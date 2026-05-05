import { analysisStreamUrl } from '../api'

export default function AnalysisStreamPage() {
  return (
    <section className="stream-page">
      <div className="stream-stage">
        <img src={analysisStreamUrl()} alt="Live Goalytics detection stream" />
      </div>
    </section>
  )
}
