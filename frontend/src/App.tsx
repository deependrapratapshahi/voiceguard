import { BrowserRouter, Routes, Route } from "react-router-dom";
import AppShell from "./components/AppShell";
import Dashboard from "./pages/Dashboard";
import LiveCallMonitoring from "./pages/LiveCallMonitoring";
import CallDetails from "./pages/CallDetails";
import SpeakerRegistry from "./pages/SpeakerRegistry";
import Alerts from "./pages/Alerts";
import Analytics from "./pages/Analytics";
import DemoMode from "./pages/DemoMode";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/live" element={<LiveCallMonitoring />} />
          <Route path="/calls/:callId" element={<CallDetails />} />
          <Route path="/speakers" element={<SpeakerRegistry />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/demo" element={<DemoMode />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
