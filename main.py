import dotenv
import argparse
from db import Database
from flow import create_teacher_flow

dotenv.load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate personalized teacher feedback for a student."
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Student ID in the database (e.g. student_001)"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching (default: enabled)"
    )

    parser.add_argument(
        "--max-subjects",
        type=int,
        default=10,
        help="Maximum number of subjects to assess"
    )

    parser.add_argument(
        "--max-topics",
        type=int,
        default=10,
        help="Maximum number of learning topics to generate"
    )

    args = parser.parse_args()

    # Load student data
    db = Database()
    student_data = db.get(args.student_id)

    if not student_data:
        raise ValueError(f"Student '{args.student_id}' not found in database")

    # Shared state for PocketFlow
    shared = {
        "student_data": student_data,
        "use_cache": not args.no_cache,
        "max_subjects": args.max_subjects,
        "max_topics": args.max_topics,
    }

    print(f"🎓 Generating teacher feedback for: {student_data.get('Full Name')}")
    print(f"LLM caching: {'Disabled' if args.no_cache else 'Enabled'}")

    # Create and run flow
    teacher_flow = create_teacher_flow()
    teacher_flow.run(shared)

    # Output result
    print("\n" + "=" * 60)
    print(shared["teacher_conclusion"])
    print("=" * 60)


if __name__ == "__main__":
    main()
