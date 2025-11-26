import React, { useState } from "react";
import { Box, Button, Card, Typography, IconButton } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import Results from "./ResultsBox";

export default function UploadImages() {
  const [images, setImages] = useState<File[]>([]);
  const [showResults, setShowResults] = useState(false);

  //Code for results here
  const [results, setResults] = useState<
    { disease: string; probability: number }[]
  >([]);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setImages(Array.from(e.target.files));
    }
  };

  //Code for results here
  const handleResults = () => {
    const mockResults = [
      { disease: "Glaucoma", probability: 75 },
      { disease: "Cataract", probability: 20 },
      { disease: "Macular Degeneration", probability: 50 },
      { disease: "Diabetic Retinopathy", probability: 10 },
    ];
    setResults(mockResults);
    setShowResults(true);
  };

  const handleRemove = (index: number) => {
    setImages(images.filter((_, i) => i !== index));
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        bgcolor: "#f5f5f5",
      }}
    >
      <Typography variant="h4" fontWeight={700} sx={{ mb: 3 }}>
        Upload Images
      </Typography>

      <Card
        sx={{
          width: "700px",
          height: "500px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "flex-start",
          bgcolor: "#f9f9f9",
          p: 2,
        }}
      >
        {/* Image Upload */}
        <Box
          sx={{
            width: "100%",
            height: "60%",
            textAlign: "center",
            border: "2px dashed #73c2ff",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            mb: 2,
          }}
        >
          <Typography variant="body1" color="text.secondary">
            Click to upload images
          </Typography>
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleUpload}
            style={{ display: "none" }}
            id="upload-input"
          />
          <label htmlFor="upload-input" style={{ cursor: "pointer" }}>
            <Button variant="outlined" sx={{ mt: 2 }} component="span">
              Choose Files
            </Button>
          </label>
        </Box>

        {/* Uploaded Images List */}
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            width: "100%",
            height: "30%",
            mb: 2,
            overflowY: "auto",
            maxHeight: 150,
          }}
        >
          {images.map((img, index) => (
            <Card
              key={index}
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                p: 1,
                mx: 2,
                height: 60,
                minHeight: 60,
                maxHeight: 60,
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <img
                  src={URL.createObjectURL(img)}
                  alt={img.name}
                  style={{
                    width: 50,
                    height: 50,
                    objectFit: "cover",
                    borderRadius: 4,
                  }}
                />
                <Typography variant="body2" color="text.secondary">
                  {img.name}
                </Typography>
              </Box>
              <IconButton onClick={() => handleRemove(index)}>
                <CloseIcon />
              </IconButton>
            </Card>
          ))}
        </Box>

        {/* Confirm Button */}
        <Box sx={{ width: "100%", textAlign: "center" }}>
          <Button
            variant="contained"
            color="primary"
            sx={{ width: 200 }}
            onClick={handleResults}
            disabled={images.length === 0}
          >
            Confirm and Run
          </Button>
        </Box>
      </Card>

      {/* Results Window */}
      <Results
        open={showResults}
        onClose={() => setShowResults(false)}
        results={results}
      />
    </Box>
  );
}
