import os
import re
import yaml
from pocketflow import Node, BatchNode
from utils.crawl_github_files import crawl_github_files
from utils.call_llm import call_llm
from utils.crawl_local_files import crawl_local_files
from db import Database
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics




# Node 1 - AppriseStudentGrades - Results of person
# --------------------------------------------------------
class AssessStudentLevel(Node):
    """
    Node: AssessStudentLevel
    Purpose: Evaluate student's knowledge across subjects
    and generate a structured profile.
    """

    def prep(self, shared):
        student_data = shared["student_data"]  # dict from Database['data']
        use_cache = shared.get("use_cache", True)
        max_subjects = shared.get("max_subjects", 10)
        return student_data, use_cache, max_subjects

    def exec(self, prep_res):
        student_data, use_cache, max_subjects = prep_res
        print(f"Assessing knowledge level for {student_data.get('Full Name', 'Unknown')}...")

        prompt = f"""
You are an experienced school teacher AI. The data you received 
 contains school grades for subjects (highest score is 5), class number,
 and student biography.

Student Data:
{student_data}

For EACH subject (up to {max_subjects}):
1. Assign a knowledge level: Very Low, Average, Above Average, High.
2. Provide reasoning in 1-3 sentences.
3. Identify main strengths and gaps.

Output STRICTLY in YAML format:

```yaml
student_profile:
  subjects:
    - name: ""
      level: ""
      reasoning: |
        ...
      strengths:
        - ""
      gaps:
        - ""
```"""

        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))

        # --- Extract YAML safely ---
        match = re.search(r"```yaml(.*?)```", response, re.DOTALL)
        if not match:
            raise ValueError("No YAML block found in LLM output")
        yaml_str = match.group(1).strip()

        profile = yaml.safe_load(yaml_str)
        if "student_profile" not in profile:
            raise ValueError("Missing 'student_profile' key in LLM output.")
        return profile

    def post(self, shared, prep_res, exec_res):
        shared["student_profile"] = exec_res
        print("Student profile stored in shared['student_profile'].")




# Node 2 - PrioritizeSubjects - Generate learning priority list(of subjects)
# --------------------------------------------------------
class PrioritizeSubjects(Node):
    """
    Node: PrioritizeSubjects
    Purpose: Create a ranked list of subjects for a student
    based on their knowledge level and gaps.
    """

    def prep(self, shared):
        student_profile = shared.get("student_profile")
        if not student_profile:
            raise ValueError("Missing 'student_profile' in shared data")
        use_cache = shared.get("use_cache", True)
        return student_profile, use_cache

    def exec(self, prep_res):
        student_profile, use_cache = prep_res
        print("Prioritizing subjects based on student profile...")

        prompt = f"""
You are an AI educational planner. You received a student's profile
with subjects, knowledge levels, strengths, and gaps:

{student_profile}

Task:
1. Rank the subjects from highest priority (needs most attention) to lowest.
2. Take into account:
   - Knowledge levels: Very Low → High (Very Low = highest priority)
   - Gaps: More gaps = higher priority
   - Strengths: Should not reduce priority if gaps exist
3. Provide reasoning for the order in 1-3 sentences.

Output STRICTLY in YAML format:

```yaml
learning_priority:
  - subject: ""
    priority: 1
    reasoning: |
      ...
```"""

        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))

        # --- Extract YAML safely ---
        import re
        match = re.search(r"```yaml(.*?)```", response, re.DOTALL)
        if not match:
            raise ValueError("No YAML block found in LLM output")
        yaml_str = match.group(1).strip()

        import yaml
        priority_list = yaml.safe_load(yaml_str)

        if "learning_priority" not in priority_list or not isinstance(priority_list["learning_priority"], list):
            raise ValueError("Missing or invalid 'learning_priority' in LLM output.")

        return priority_list

    def post(self, shared, prep_res, exec_res):
        shared["learning_priority"] = exec_res
        print("Learning priority stored in shared['learning_priority'].")

# Node 3 - KnowledgeToDiscover - Lists a theme and topic to learn
# --------------------------------------------------------
class KnowledgeToDiscover(Node):

    def prep(self, shared):
        student_profile = shared.get("student_profile")
        learning_priority = shared.get("learning_priority")
        if not student_profile or not learning_priority:
            raise ValueError("Missing 'student_profile' or 'learning_priority' in shared data")
        use_cache = shared.get("use_cache", True)
        max_topics = shared.get("max_topics", 10)
        return student_profile, learning_priority, use_cache, max_topics

    def exec(self, prep_res):
        student_profile, learning_priority, use_cache, max_topics = prep_res
        print("Generating topics and subtopics to discover...")

        prompt = f"""
You are an AI tutor. You received the following data:

1. Student profile with subjects, knowledge levels (Very Low / Average / Above Average / High),
   strengths, and gaps:
{student_profile}

2. Ranked learning priority of subjects (highest priority = needs most attention):
{learning_priority}

Task:
- Generate a clear study plan for the student.
- Focus ONLY on subjects with:
  * middle-level knowledge (Average / Above Average)
  * notable gaps
- For each such subject, create:
  1. Main topic name (`topic`)
  2. Source of topic suggestion (`based_from`): e.g., "class middle level" or "identified gaps"
  3. 2-3 practical examples (`examples`) the student can practice
  4. 2-5 subtopics (`subtopics`) with their source (`based_from`), highlighting gaps or weaknesses

Output STRICTLY in YAML format, as a list of main topics:

```yaml
knowledge_to_discover:
  - topic: "Main Topic Name"
    based_from: "class middle level / identified gaps"
    examples:
      - "Practical Example 1"
      - "Practical Example 2"
    subtopics:
      - name: "Subtopic 1"
        based_from: "gap or weakness"
      - name: "Subtopic 2"
        based_from: "gap or weakness"
# Repeat up to 10 main topics```
"""
        response = call_llm(prompt, use_cache=(use_cache and self.cur_retry == 0))

        # Extract YAML safely
        match = re.search(r"```yaml(.*?)```", response, re.DOTALL)
        if not match:
            raise ValueError("No YAML block found in LLM output")
        yaml_str = match.group(1).strip()
        knowledge = yaml.safe_load(yaml_str)

        if "knowledge_to_discover" not in knowledge or not isinstance(knowledge["knowledge_to_discover"], list):
            raise ValueError("Missing or invalid 'knowledge_to_discover' key in LLM output.")

        return knowledge

    def post(self, shared, prep_res, exec_res):
            shared["knowledge_to_discover"] = exec_res
            print("Knowledge topics and subtopics stored in shared['knowledge_to_discover'].")


class FinalTeacherConclusion(Node):
    """
    Final Node:
    Generates a complete, human-readable teacher conclusion
    and saves it as a set of Markdown files.
    """

    def prep(self, shared):
        return (
            shared["student_data"],
            shared["student_profile"],
            shared["learning_priority"],
            shared["knowledge_to_discover"],
            shared.get("output_dir", "output"),
            shared.get("use_cache", True),
        )

    def exec(self, prep_res):
        student_data, profile, priority, plan, output_dir, use_cache = prep_res

        name = student_data.get("Full Name", "ученик")
        grade = student_data.get("Class", "N/A")

        # ---- Подробный промпт на русском ----
        prompt = f"""
Вы — заботливый и опытный школьный учитель.

Ваша задача — составить подробный и полезный итоговый отзыв для ученика. 
Текст будет читаться учеником и родителями.

Имя ученика: {name}
Класс: {grade}

Профиль ученика (уровни, сильные стороны, пробелы):
{profile}

Приоритеты в обучении:
{priority}

Учебный план:
{plan}

Напишите подробное, структурированное заключение на русском языке в формате Markdown. 
Разделите текст на отдельные логические блоки, каждый с заголовком (например, ### Общая оценка, ### Предметы, требующие внимания и т.д.).
"""

        # ---- Вызов LLM ----
        text = call_llm(prompt, use_cache=(use_cache and getattr(self, "cur_retry", 0) == 0))

        # ---------- MARKDOWN GENERATION ----------
        os.makedirs(output_dir, exist_ok=True)
        safe_name = re.sub(r"[^\w]+", "_", name.lower())

        # Разделим текст на блоки по заголовкам уровня 3 (###)
        blocks = text.split("\n### ")
        file_paths = []

        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            if i == 0 and not block.startswith("### "):
                block = "### " + block  # для первого блока, если нет ###

            md_file = os.path.join(output_dir, f"{safe_name}_part{i + 1}.md")
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(block)
            file_paths.append(md_file)

        # Создаём index.md с включением всех частей
        index_file = os.path.join(output_dir, f"{safe_name}_index.md")
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(f"# Итоговое заключение учителя для {name}\n\n")
            for path in file_paths:
                filename = os.path.basename(path)
                f.write(f"- [{filename}]({filename})\n")

        return {
            "text": text,
            "md_files": file_paths,
            "index_md": index_file
        }

    def post(self, shared, prep_res, exec_res):
        shared["teacher_conclusion"] = exec_res["text"]
        shared["teacher_conclusion_md_files"] = exec_res["md_files"]
        shared["teacher_conclusion_index_md"] = exec_res["index_md"]
        print(f"📄 Teacher conclusion saved as Markdown files: {exec_res['index_md']}")
