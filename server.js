const express = require("express");
const { getClinicalTrials, ragQuery } = require(".");
const cors = require("cors");  // <-- import cors


const app = express();
app.use(cors());
app.use(express.json());


app.listen(process.env.PORT || 4000, () => {
  console.log("Server is running on port 3000");
});

app.get("/", (req, res) => {
  res.send("Hello World!");
});


app.post("/clinical-trials",getClinicalTrials)
app.post("/rag-query",ragQuery)
