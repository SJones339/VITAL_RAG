# Project VITAL — LLM + RAG Testing Tutorial (Student Starter)

**CSCI3364 — Software Testing, Quality and Security | Boston College**  

In this tutorial you will build a local LLM chatbot, extend it with retrieval-augmented generation (RAG), and design tests that distinguish retrieval failures from generation failures.

## Before class

1. Install [Ollama](https://ollama.com/download).
2. Run `ollama pull qwen2.5:1.5b` (approximately 1 GB).
3. Check that `ollama run qwen2.5:1.5b` answers a question; type `/bye` to exit.
4. Install Python 3.10 or later.

## Run the baseline chatbot

```bash
git clone https://github.com/CSCI3364-Fall2026/VITAL-RAG-Starter.git
cd VITAL-RAG-Starter
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The baseline chatbot works without RAG. If Ollama is unavailable, check that its service is running and the model is downloaded.

## Lab schedule

| Activity |
| --- | --- |
| LLMs, hallucinations, RAG and test oracles |
| Run chatbot and establish baseline responses |
| Load policy documents and implement TF-IDF retrieval |
| Add source context to model calls and a RAG toggle |
| Design and run automated tests; compare answers |
| Interpret results and report conclusions |

## Tasks (write the code yourself)

1. **Baseline**: Ask three policy questions before implementing RAG. Record the answers.
2. **TODO 1 — load documents** in `rag.py`. Return sorted `(filename, text)` tuples for `.txt` files.
3. **TODO 2 — retrieve** in `rag.py`. Use `TfidfVectorizer` and cosine similarity to rank documents. Return at most `k` `(filename, text, score)` tuples. Exclude zero-score results and break ties by filename.
4. **TODO 3 — format context** in `rag.py`. Prefix each selected document with `[Source: filename]`.
5. **TODO 4 — integrate RAG** in `app.py`. Add a sidebar **Enable RAG** toggle and send context to `ask_llm(question, context)` when enabled. Display retrieved filenames to the user.
6. **Test and improve**: run `pytest -q`, add at least one test, investigate a failure, make one improvement, and rerun.

The included `tests/test_retrieval.py` intentionally **fails initially** because the retrieval functions are incomplete. Passing those tests is a milestone, not a guarantee that generated answers are correct.

## Suggested questions

- How many hours before an appointment must I cancel?
- Can I request a same-day appointment?
- How are laboratory results communicated?
- Does the clinic provide free transportation? (The documents do not answer this.)
- What happens when I ask about an unsupported policy?

## Evidence to submit

For at least four questions, record: (a) question, (b) expected source, (c) retrieved filenames, (d) response without RAG, (e) response with RAG, and (f) whether the response is **supported by the source**. Explain one failure, your change, and the outcome of rerunning the test. Distinguish failures in retrieval from failures in generation.

## Project writeup

Fill in the blanks after running the chatbot (RAG off, then RAG on for each question).

| (a) Question                                        | (b) Expected source | (c) Retrieved filenames                                  | (d) Response without RAG                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | (e) Response with RAG                                                       | (f) Supported by source? |
| --------------------------------------------------- | ------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------ |
| How many hours before an appointment must I cancel? | `cancellations.txt` | Sources: cancellations.txt, scheduling.txt, visitors.txt | I'm sorry, but I can't answer this question as it doesn't provide enough context. Could you please provide more information or clarify the question?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | You must cancel an appointment at least 18 hours before the scheduled time. | **Yes** — 18-hour rule is in `cancellations.txt` |
| Can I request a same-day appointment?               | `scheduling.txt`    | Sources: scheduling.txt, portal.txt                      | I'm sorry, but I can't assist with that.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Yes, you can request a same-day appointment, but it must be available.      | **Yes** — matches `scheduling.txt` |
| How are laboratory results communicated?            | `lab_results.txt`   | Sources: lab_results.txt, portal.txt                     | Laboratory results are typically communicated in several ways, depending on the context and the nature of the results. Here are some common methods:- **Written Reports**: Laboratory results are often reported in written form, either as a report or as part of a larger report. This can be sent to the patient, the referring physician, or the patient's insurance company.- **Email or Electronic Health Records (EHR)**: Results can be sent electronically via email or integrated into electronic health records (EHR) systems. This is particularly useful for sharing results with multiple healthcare providers.- **Phone Calls**: Results can be communicated over the phone, especially if the results are urgent or require immediate action.- **In-Person Consultations**: In some cases, results may be discussed in person, either at the patient's request or as part of a follow-up appointment.- **Notifications**: Some laboratories may send notifications via text message or mobile app, especially for results that are not immediately actionable.- **Patient Education**: In some cases, the laboratory may provide educational materials or information about the results, especially if the results are complex or require further explanation.- **Follow-Up Appointments**: Results may be discussed during follow-up appointments, either in person or via telehealth.The method of communication is often determined by the urgency of the results, the preferences of the patient or healthcare provider, and the specific requirements of the laboratory and the healthcare system. | Laboratory results are communicated through the secure patient portal.      | **Yes** — matches `lab_results.txt` |
| Does the clinic provide free transportation?        | none (not in docs)  | Sources: billing.txt, lab_results.txt, scheduling.txt    | I'm sorry, but I can't answer this question because I don't have any specific information about a particular clinic. The answer would depend on the clinic's policies and the location. If you know the name of the clinic or the location, you can search online or contact them directly to find out if they provide free transportation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | No, the clinic does not provide free transportation.                        | **No** — docs never say this; model invented a denial (generation failure). Prefer: not enough info in documents |

### Failure, change, and retest

- **Failure:** Using `TfidfVectorizer(stop_words="english")` made the cancel question retrieve `scheduling.txt` and `visitors.txt` ahead of `cancellations.txt` (retrieval failure — wrong ranking).
- **Change:** Removed English stop words (plain `TfidfVectorizer()`). Also increased app retrieval to `k=3`. Added tests for lab-results retrieval and zero-score / filename tie-break.
- **Outcome of rerunning `pytest -q`:** 6 passed.

### Retrieval vs generation

- **Retrieval failure means:** The wrong documents are returned (or the right one is missing) in Sources, for example stop words ranking cancel below scheduling, or transportation retrieving unrelated billing,lab,scheduling, etc. files.
- **Generation failure means:** Sources may be fine (or empty of the needed fact), but the model invents or generates incorrect content, for example answering “No” for free transportation when the docs never say that.

**Security and privacy:** The six clinic policy files are fictional educational materials. Never enter real patient information or protected health information. Treat retrieved documents as untrusted data.

[Project VITAL course repository](https://github.com/CSCI3364-Fall2026/Project_VITAL)
