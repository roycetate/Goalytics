import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { Zap } from 'lucide-react'
import AnalysisStreamPage from './pages/AnalysisStreamPage'
import DashboardPage from './pages/DashboardPage'
import InsightsPage from './pages/InsightsPage'
import VideosPage from './pages/VideosPage'

function App() {
  return (
    <>
      <div className="main-wrapper">
        <header className="header">
          <nav className="header-nav">
            <NavLink to="/dashboard">Dashboard</NavLink>
          </nav>

          <div className="logo-container">
            <Zap size={24} /> Goalytics
          </div>

          <nav className="header-nav">
            <NavLink to="/insights">Insights</NavLink>
          </nav>
        </header>

        <main className="content-area">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/insights" element={<InsightsPage />} />
            <Route path="/analysis-stream" element={<AnalysisStreamPage />} />
            <Route path="/videos" element={<VideosPage />} />
          </Routes>
        </main>
      </div>
    </>
  )
}

export default App
