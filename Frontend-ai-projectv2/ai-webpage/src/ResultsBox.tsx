import {
  Box,
  Card,
  Typography,
  Dialog,
  IconButton,
  DialogTitle,
  DialogContent,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

export default function Results({
  open,
  onClose,
  results,
}: {
  open: boolean;
  onClose: () => void;
  results: { disease: string; probability: number }[];
}) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        Diagnosis Results
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{ position: "absolute", right: 8, top: 8 }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {results.map((res, index) => (
          <Typography key={index} variant="h6" sx={{ mb: 1 }}>
            {res.disease}: {res.probability}%
          </Typography>
        ))}
      </DialogContent>
    </Dialog>
  );
}
