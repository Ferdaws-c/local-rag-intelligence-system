# 🎬 Presentation Video Script (Plain English Version)
## Ferdaws Qaem — Local AI Assistant
### Final Presentation Guide

---

## ⏱️ Target Length: 5–8 minutes

---

## SECTION 1 — Introduction (0:00 – 0:45)

**[What to do on screen: Show your GitHub repository page]**

**[What to say:]**
> "Hi everyone, my name is Ferdaws Qaem, and this is my final project for the Local AI summer school program. 
>
> I built a smart AI assistant that acts like my personal representative. It can answer questions about my academic background, my work experience, and my projects. 
>
> But what makes this project special is privacy. **Everything runs completely offline on my own computer.** No data is sent to the internet, and I don't use any paid cloud services. The AI lives right here on my laptop."

---

## SECTION 2 — The Problem (0:45 – 1:30)

**[What to do on screen: Show a folder with your 5 text documents (transcript, profile, etc.)]**

**[What to say:]**
> "Why did I build this? 
>
> Imagine a recruiter or a teacher wanting to know about my background. Normally, they would have to read through 5 different documents: my official transcript, my resume, my project details, and so on. That takes a lot of time.
>
> With my app, they can just type a question in plain English, and the AI instantly reads all my documents and gives them an exact answer in seconds.
>
> In the tech world, this method is called **RAG** (Retrieval-Augmented Generation), which basically means 'Search first, then Answer'."

---

## SECTION 3 — How It Works in Plain English (1:30 – 2:30)

**[What to do on screen: Open your app in the browser so it's ready to use]**

**[What to say:]**
> "Before I show you the app, let me explain how it works in 3 simple steps:
>
> **Step 1: Reading.** The app reads my 5 text documents and chops them up into small paragraphs. It stores these paragraphs in a tiny, local database.
>
> **Step 2: Searching.** When you ask a question, the app searches that database to find the top 5 paragraphs that are most related to your question.
>
> **Step 3: Answering.** Finally, the app takes your question and those 5 paragraphs, and gives them to the local AI. It gives the AI a very strict rule: *'Answer the user's question using ONLY the information in these paragraphs. If the answer isn't there, say you don't know.'* 
>
> Because of this rule, the AI won't lie or make things up."

---

## SECTION 4 — Live Demo (2:30 – 5:30)

**[What to do on screen: Show the Streamlit chat interface]**

**[What to say:]**
> "Let's see it in action."

### Demo Query 1 — A basic question ✅
**[Type in the chat box:]** `What is Ferdaws's student ID?`

> "First, I'll ask about my student ID. Watch how the AI types out the answer in real-time."

**[Wait for answer to finish, then point to the 'Sources' panel at the bottom]**

> "Not only does it answer correctly, but if you look down here at the 'Sources' panel, it shows us exactly which document it read to find that answer. It pulled this right from my `official_transcript.txt`."

---

### Demo Query 2 — An experience question ✅
**[Type:]** `Has Ferdaws completed an Erasmus program?`

> "Now let's ask a question about my academic experience."

**[Wait for answer]**

> "Again, it searches my documents, finds the section about my Erasmus exchange program, and gives a clear answer."

---

### Demo Query 3 — The Trick Question ✅
**[Type:]** `Who is the president of the United States?`

> "Now, this is the most important test. I'm going to ask it a random trick question that has nothing to do with me."

**[Wait for answer — it should say 'I don't have that information']**

> "As you can see, the assistant refuses to answer. It doesn't guess, and it doesn't make things up. It simply says 'I don't have that information.' This proves that the AI is safely restricted to only my personal documents."

---

## SECTION 5 — Testing Results (5:30 – 6:30)

**[What to do on screen: Open your code editor and show the `test_suite.py` file or the test results text file]**

**[What to say:]**
> "To make sure the app works perfectly, I wrote an automated test script. 
> 
> The script automatically asked the AI 12 different questions: some about my profile, some trick questions, and some confusing edge cases (like just typing a question mark).
>
> I'm happy to report that the AI scored a 100% pass rate. It answered the profile questions correctly and safely refused the trick questions every single time."

---

## SECTION 6 — Lessons Learned (6:30 – 7:30)

**[What to do on screen: Return to your app or GitHub page]**

**[What to say:]**
> "To wrap up, here are the two biggest things I learned from this project:
>
> **Number one: You have to be bossy with AI.** If you just tell an AI 'answer this question', it will sometimes get creative and invent details. I had to use strict words like 'Do NOT use outside knowledge' to force it to stick to the facts.
>
> **Number two: Running AI offline is amazing.** I ran all of this on my own laptop's hardware. It's fast, it's completely private, and it shows that you don't always need expensive cloud servers to build powerful AI tools."

---

## SECTION 7 — Closing (7:30 – 8:00)

**[What to do on screen: Show your GitHub repository page one last time]**

**[What to say:]**
> "All the code for this project is open source and available on my GitHub. The instructions are included so anyone can download it and run it on their own machine.
>
> Thank you for watching!"
