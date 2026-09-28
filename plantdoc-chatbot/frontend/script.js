console.log("✅ NEW script.js loaded");

async function submitData() {
  const textInput = document.getElementById("textInput");
  const imageInput = document.getElementById("imageInput");
  const chatBox = document.getElementById("chat-box");

  const text = textInput.value.trim();
  const image = imageInput.files[0];

  if (!text && !image) return;

  /* =========================
     USER MESSAGE BUBBLE
  ========================== */
  const userMsg = document.createElement("div");
  userMsg.className = "user-message";

  if (text && image) {
    userMsg.innerText = `📝 ${text}\n📷 Image uploaded`;
  } else if (text) {
    userMsg.innerText = `📝 ${text}`;
  } else if (image) {
    userMsg.innerText = `📷 Image uploaded`;
  }

  chatBox.appendChild(userMsg);
  chatBox.scrollTop = chatBox.scrollHeight;

  textInput.value = "";
  imageInput.value = "";

  /* =========================
     BOT "ANALYZING" MESSAGE
  ========================== */
  const botMsg = document.createElement("div");
  botMsg.className = "bot-message";
  botMsg.innerText = "🤖 Analyzing...";
  chatBox.appendChild(botMsg);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    /* =========================
       SEND TEXT + IMAGE TOGETHER
    ========================== */
    const formData = new FormData();
    if (text) formData.append("text", text);
    if (image) formData.append("image", image);

    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      body: formData
    });

    const rawText = await response.text();
    console.log("🔵 Raw response:", rawText);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${rawText}`);
    }

    const data = JSON.parse(rawText);

    /* =========================
       DISPLAY RESPONSE
    ========================== */
    if (data.mode === "image + text") {
      botMsg.innerHTML = `
        🧠 <b>Image + Text analyzed</b><br>
        🌿 <b>Disease:</b> ${data.final_disease}<br>
        📊 <b>Confidence:</b> ${(data.image_confidence * 100).toFixed(2)}%<br>
      `;
    }
    else if (data.disease) {
      botMsg.innerHTML = `
        🌿 <b>Disease:</b> ${data.disease}<br>
        📊 <b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%
      `;
    }
    else {
      botMsg.innerText = data.message || "⚠️ Unexpected response";
    }

  } catch (err) {
    console.error("❌ Frontend error:", err);
    botMsg.innerText = "❌ Server error. Check console for details.";
  }

  chatBox.scrollTop = chatBox.scrollHeight;
}

/* =========================
   ENTER KEY SUBMIT
========================== */
document.getElementById("textInput").addEventListener("keydown", function (e) {
  if (e.key === "Enter") {
    e.preventDefault();
    submitData();
  }
});
