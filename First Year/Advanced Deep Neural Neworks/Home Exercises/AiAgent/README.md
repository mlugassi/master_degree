# 📝 TaskMaster – AI-Powered Task Manager

**TaskMaster** הוא סוכן אינטראקטיבי מבוסס LangGraph המאפשר ניהול משימות בשפת אנוש. הסוכן מנתח פקודות טקסטואליות, מוסיף משימות, מסמן כהושלמו ומציג את רשימת המשימות – והכל באינטראקציה ישירה עם המשתמש.

---

## 🚀 הרצה

יש להריץ את הקובץ הראשי:
```bash
python taskmaster.py
```

עם ההפעלה, יופיע מסך קבלת פנים וניתן להזין פקודות לניהול משימות.

---

## 💡 פקודות נתמכות

- `add_task name:"<task name>" description:"<optional description>" priority:"<optional priority>"`
  - דוגמה: `add_task name:"Buy milk" description:"2 liters of milk" priority:"high"`
- `mark_complete <task ID>`
  - דוגמה: `mark_complete 0`
- `get_tasks [all|completed|pending]`
  - דוגמה: `get_tasks completed`
- `exit` / `quit` / `bye` – יציאה מהמערכת

---

## 🧠 מבנה הפרויקט

### 1. `Task` – מבנה המשימה
```python
class Task(TypedDict):
    id: str
    name: str
    description: str
    priority: Literal["low", "medium", "high"]
    completed: bool
```

### 2. `GraphState` – מצב כולל של הסוכן
כולל רשימת משימות, מזהה ID הבא, סטטוס סיום, והודעות ההיסטוריה.

---

## 🧩 רכיבי LangGraph

### Nodes:
- **init** – אתחול המצב הראשוני
- **router** – זיהוי סוג הפקודה
- **add** – הוספת משימה
- **complete** – סימון משימה כהושלמה
- **list** – הצגת משימות
- **wait** – הצגת שאלה המשך
- **unrecognize** – מענה במידה והפקודה לא מזוהה
- **finish** – סיום הסשן

### Edges:
- המעברים בין הצמתים תלויים בפלט הפונקציה `route_user_input`.

---

## 🛠 הסברים על קוד מרכזי

### הוספת משימה
```python
re.search(r'(?:name):\s*"([^"]+)"', input)
```
מזהה את שם המשימה מתוך הפקודה. בדומה לכך, מאתרים גם תיאור (`desc`) ועדיפות (`prior`). ברירת מחדל לעדיפות היא `"medium"`.

### השלמת משימה
```python
if task["id"] == task_id:
    task["completed"] = True
```
הסוכן סורק את רשימת המשימות, מסמן כ־`completed = True` לפי ה־ID.

### הצגת משימות
מתבצע סינון לפי `all`, `completed` או `pending`.

---

## 📦 דרישות

- Python 3.9+
- ספריות:
  - `langchain-core`
  - `langgraph`

---

## ✨ יתרונות

- שפת פקודה טבעית
- זרימה שיחתית עם Feedback
- ניהול זיכרון פנימי של משימות
- עיצוב תרחישי קצה (פקודות שגויות וכו')
