import { BrowserRouter, Route, Routes } from "react-router-dom";

import { KnowledgeBaseDetailPage } from "./routes/KnowledgeBaseDetailPage";
import { KnowledgeBaseListPage } from "./routes/KnowledgeBaseListPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<KnowledgeBaseListPage />} />
      <Route
        path="/knowledge-bases/:kbId"
        element={<KnowledgeBaseDetailPage />}
      />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
