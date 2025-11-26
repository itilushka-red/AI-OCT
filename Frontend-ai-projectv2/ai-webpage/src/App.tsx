import React from "react";
import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadImages from "./UploadImages";

function App() {
  return (
    <React.StrictMode>
      <Router>
        <Routes>
          <Route path="/" element={<UploadImages />} />
        </Routes>
      </Router>
    </React.StrictMode>
  );
}

export default App;
