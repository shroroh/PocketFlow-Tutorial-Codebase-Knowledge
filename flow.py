from pocketflow import Flow

from nodes import (
    AssessStudentLevel,
    PrioritizeSubjects,
    KnowledgeToDiscover,
    FinalTeacherConclusion,
)

def create_teacher_flow():
    """Creates and returns the Teacher AI Agent flow."""

    assess_student = AssessStudentLevel(max_retries=3, wait=10)
    prioritize_subjects = PrioritizeSubjects(max_retries=3, wait=10)
    knowledge_to_discover = KnowledgeToDiscover(max_retries=3, wait=10)
    final_conclusion = FinalTeacherConclusion()

    # Connect nodes
    assess_student >> prioritize_subjects
    prioritize_subjects >> knowledge_to_discover
    knowledge_to_discover >> final_conclusion

    # Create flow
    teacher_flow = Flow(start=assess_student)

    return teacher_flow
