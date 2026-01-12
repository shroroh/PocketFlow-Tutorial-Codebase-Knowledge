# db.py
# просто файл пустышка со структурой данных, заполняемых из БД, или как представленно здесь...
class Database:
    def __init__(self):
        self.data = [
            {
                "Full Name": "Иван Иванов",
                "Login": "ivan123", # связка логин-пароль для авторизации
                "Password": "ivan",
                "Marks and exams": { # Оценки
                    "Math": [5, 4, 3, 4, 5],
                    "Physics": [5, 4, 3, 4, 5],
                    "English": [5, 4, 3, 4, 5],
                    "History": [5, 4, 3, 4, 5],
                    "Informatics": [3, 4, 3, 4, 3],
                    "Russian language": [2, 4, 3, 4, 2]
                },
                "Personal": {
                    "Class": 9,
                    "Bio": "Люблю математику и физику, играю в шахматы на уровне любителя."
                }
            },
            {
                "Full Name": "Иван Петров",
                "Login": "ivan_petrov",
                "Password": "ivan",
                "Marks and exams": {
                    "Math": [5, 4, 3, 4, 5],
                    "Physics": [5, 4, 3, 4, 5],
                    "English": [5, 4, 3, 4, 5],
                    "History": [5, 4, 3, 4, 5],
                    "Informatics": [3, 4, 3, 4, 3],
                    "Russian language": [2, 4, 3, 4, 2]
                },
                "Personal": {
                    "Class": 9,
                    "Bio": "Люблю информатику, играю в шахматы на уровне гроссмейстера."
                }
            },
            {
                "Full Name": "Мария Петрова",
                "Login": "maria123",
                "Password": "maria",
                "Marks and exams": {
                    "Math": [5, 3, 4, 5],
                    "Physics": [2, 4, 2, 4, 2],
                    "English": [5, 2, 3, 4, 5],
                    "History": [5, 4, 3, 4, 2],
                    "Informatics": [3, 5, 5, 4, 5],
                    "Russian language": [2, 4, 5, 4, 2]
                },
                "Personal": {
                    "Class": 9,
                    "Bio": "Мне нравится литература и история, я посещаю кружок по драме!"
                }
            }
        ]

    def get(self, login: str):
        """
        Get student by login (used as student_id)
        """
        for student in self.data:
            if student.get("Login") == login:
                return self._normalize(student)
        return None

    def _normalize(self, student: dict) -> dict:
        """
        Normalize structure for AI nodes
        """
        return {
            "Full Name": student["Full Name"],
            "Class": student["Personal"]["Class"],
            "Bio": student["Personal"]["Bio"],
            "Marks": student["Marks and exams"]
        }
