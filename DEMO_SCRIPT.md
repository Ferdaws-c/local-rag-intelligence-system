# 🎬 Demo Video Script
## Ferdaws Qaem — Local RAG Assistant
### Final Presentation Guide

---

## ⏱️ Target Length: 5–8 minutes

---

## SECTION 1 — Introduction (0:00 – 0:45)

**[Screen: Show the GitHub repository page]**

> "Hi, my name is Ferdaws Qaem, and this is my final project for the Local AI with Microsoft Foundry Local summer school program.
>
> I built a fully offline AI-powered Q&A assistant for my academic and professional profile.
> What makes this project special is that **everything runs on my local machine** — no cloud services, no API keys, no internet connection needed for the AI part.
> The entire AI pipeline — from understanding your question to generating an answer — happens right here on this computer."

---

## SECTION 2 — Problem Statement (0:45 – 1:30)

**[Screen: Keep GitHub open or switch to a simple slide/notes]**

> "So what problem does this solve?
>
> Imagine you're a recruiter or hiring manager reviewing my profile. I have 5 documents — my profile overview, official transcript, work experience, projects, and Microsoft AI Innovators details.
>
> Without AI, you'd have to search through all of these manually to find specific details about my background.
> With this assistant, they just **ask a question in plain English** and get a precise, cited answer in seconds — all running privately on their own device.
>
> This is what's called **Retrieval-Augmented Generation**, or RAG."

---

## SECTION 3 — Architecture Explanation (1:30 – 2:30)

**[Screen: Open README.md on GitHub and point to the architecture diagram]**

> "Let me quickly explain how it works under the hood — there are 4 steps.
>
> **Step 1 — Ingestion**: When we first set up the app, a script called `ingest.py` reads all 5 documents, splits them into paragraph-level chunks, and converts each chunk into a numerical vector called an **embedding** using a local AI model. These vectors are stored in a lightweight SQLite database.
>
> **Step 2 — Retrieval**: When you ask a question, the app converts your question into an embedding too, then compares it mathematically to every stored chunk using **cosine similarity** — essentially measuring 'how similar is this question to each paragraph?' The top 5 most relevant paragraphs are selected.
>
> **Step 3 — Augmentation**: Those paragraphs are injected into a strict system prompt that tells the AI: 'Answer using ONLY what you see in the context below. If the answer isn't there, say you don't know.'
>
> **Step 4 — Generation**: A local large language model reads the question and the context and generates a grounded, cited answer."

---

## SECTION 4 — Live Demo (2:30 – 5:30)

**[Screen: Run `streamlit run app.py`, browser opens at localhost:8501]**

> "Let me show you the actual application."

### Demo Query 1 — In-context question ✅
**[Type in the chat box:]** `What is Ferdaws's student ID?`

> "I'll ask about my student ID. Watch how it streams the answer word by word in real time, and notice the 'Sources Retrieved' panel at the bottom — it tells us exactly which document the answer came from and how confident the retrieval was."

**[Wait for answer, then point to sources panel]**

> "You can see it pulled from `official_transcript.txt` with a high similarity score, and it correctly cited the document in its answer."

---

### Demo Query 2 — Professional question ✅
**[Type:]** `Has Ferdaws completed an Erasmus program?`

> "Now let me ask a question about my academic experience."

**[Wait for answer]**

> "It pulled from `official_transcript.txt` and gave a clear answer — again, only from the actual documents."

---

### Demo Query 3 — Out-of-context question ✅
**[Type:]** `Who is the president of the United States?`

> "Now, this is the most important test. I'll ask something completely outside the knowledge base."

**[Wait for answer — expected: 'I don't have that information']**

> "And there it is — the assistant correctly refuses to answer. It doesn't make something up, it doesn't hallucinate. It says exactly what we programmed it to say: 'I don't have that information.' This is critical for a trustworthy production system."

---

### Demo Query 4 — Model switching ⚡
**[Switch sidebar to '⚡ Fast — qwen2.5-0.5b']**

> "One more feature I'm proud of — the sidebar lets you switch between three different AI models. This Fast model is only 0.5 billion parameters — watch how quickly it responds."

**[Type:]** `What university does Ferdaws attend?`

> "About 1-2 seconds. The tradeoff is answer quality. The Balanced model takes longer but gives more nuanced responses. This lets users choose based on their hardware and patience."

---

## SECTION 5 — Testing Results (5:30 – 6:30)

**[Screen: Switch to terminal, run `python test_suite.py`, or show pre-saved test_results.txt]**

> "For Week 5, I built an automated test suite that evaluates the assistant across 12 test cases in three categories."

**[Point to results on screen]**

> "In-context questions — where the answer IS in the documents — the assistant passes consistently.
> Out-of-context questions — where the answer is NOT in the documents — it correctly falls back.
> Edge cases like typing just 'hi' or a single '?' — it handles gracefully without crashing."

---

## SECTION 6 — Lessons Learned (6:30 – 7:30)

**[Screen: Return to app or GitHub]**

> "Three key things I learned building this project.
>
> **First — chunking strategy matters enormously.** Early on, we were only retrieving 2 chunks, and we kept getting the title paragraph of the document instead of the actual content. Increasing to top_k=5 fixed retrieval quality dramatically.
>
> **Second — prompt engineering is the difference between a trustworthy AI and a hallucinating one.** Simply saying 'use only the context' isn't enough. You need to use words like 'STRICT', 'EXACTLY', and 'Do NOT use outside knowledge' to get small models to follow the rules.
>
> **Third — local AI is completely viable.** Every single model in this app runs offline on my own hardware. No data leaves this machine. For privacy-sensitive use cases like customer support, this is a genuine advantage over cloud-based solutions."

---

## SECTION 7 — Closing (7:30 – 8:00)

**[Screen: GitHub repository]**

> "The full source code is on GitHub — link in the description. The README has complete setup instructions so anyone can clone this and run it on their own machine in under 5 minutes.
>
> Thank you for watching!"

---

## 📝 Pre-Recording Checklist

- [ ] Foundry Local service is running (verify no errors in system tray)
- [ ] Run `python ingest.py` in `final_project/` to confirm DB is fresh
- [ ] Run `streamlit run app.py` and confirm it loads in browser
- [ ] Test all 3 demo queries manually BEFORE recording to confirm models are cached
- [ ] Set screen resolution to 1920×1080 for crisp recording
- [ ] Use a screen recorder (OBS Studio, Xbox Game Bar, or Loom)
- [ ] Record in a quiet room — microphone quality matters
- [ ] Do one full dry run before the final take
