import re, logging, json
from pydantic import BaseModel
from typing import Dict, Any, Optional, Type
from jb_manager_bot import AbstractFSM
from jb_manager_bot.data_models import (
    FSMOutput,
    Message,
    MessageType,
    Status,
    FSMIntent,
    TextMessage,
    ListMessage,
    ButtonMessage,
    Option,
    ImageMessage,
)
from datetime import datetime
from jb_manager_bot.parsers import Parser, OptionParser
from jb_manager_bot.parsers.utils import LLMManager

_logger = logging.getLogger(__name__)


# Variables are defined here
class TeacherCompetencyVariables(BaseModel):
    name: Optional[str] = None
    language: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[str] = None
    competency_domain: Optional[str] = None
    quiz_score: Optional[int] = None
    home_screen_option: Optional[str] = None
    question_list: Optional[list] = None
    total_num_question: int = 5
    question_count: int = 0
    current_question_id: Optional[int] = None
    user_input_param: Optional[list] = []

class TeacherCompetencyFSM(AbstractFSM):

    states = [
        "zero",
        "language_selection",
        "welcome_message_display",
        "collect_name",
        "collect_name_logic",
        "collect_subject",
        "collect_subject_logic",
        "collect_grade",
        "collect_grade_logic",
        "confirmation",
        "show_question",
        "question_answer_input",
        "answer_input_logic",
        "waiting_message",
        "calculate_score",
        "thankyou_message",
        "home_screen",
        "home_screen_logic",
        "learn_competencies",
        "assessment",
        "resources_tips",
        "help_support",
        "end",
    ]
    transitions = [
        {"source": "zero", "dest": "language_selection", "trigger": "next"},
        {"source": "language_selection", "dest": "welcome_message_display", "trigger": "next"},
        {"source": "welcome_message_display", "dest": "collect_name", "trigger": "next"},
        
        # Collecting teacher details step by step
        {"source": "collect_name", "dest": "collect_name_logic", "trigger": "next"},
        {"source": "collect_name_logic", "dest": "collect_subject", "trigger": "next"},

        {"source": "collect_subject", "dest": "collect_subject_logic", "trigger": "next"},
        {"source": "collect_subject_logic", "dest": "collect_grade", "trigger": "next"},
        {"source": "collect_grade", "dest": "collect_grade_logic", "trigger": "next"},
        {"source": "collect_grade_logic", "dest": "confirmation", "trigger": "next"},
        
        {"source": "confirmation", "dest": "home_screen", "trigger": "next"},
        
        # Navigating from home screen
        {"source": "home_screen", "dest": "home_screen_logic", "trigger": "next"},
        {"source": "home_screen_logic", "dest": "learn_competencies", "trigger": "next", "conditions": "is_learn_competencies_selected"},
        {"source": "home_screen_logic", "dest": "assessment", "trigger": "next", "conditions": "is_assessment_selected"},
        {"source": "home_screen_logic", "dest": "resources_tips", "trigger": "next", "conditions": "is_resources_tips_selected"},
        {"source": "home_screen_logic", "dest": "help_support", "trigger": "next", "conditions": "is_help_support_selected"},

        # Questions for assessment
        {"source": "assessment", "dest": "show_question", "trigger": "next"},
        {"source": "show_question", "dest": "question_answer_input", "trigger": "next"},
        {"source": "question_answer_input", "dest": "answer_input_logic", "trigger": "next"},
        {"source": "answer_input_logic", "dest": "show_question", "trigger": "next", "conditions": "is_send_another_question"},
        {"source": "answer_input_logic", "dest": "waiting_message", "trigger": "next"},
        {"source": "waiting_message", "dest": "calculate_score", "trigger": "next"},
        {"source": "calculate_score", "dest": "thankyou_message", "trigger": "next"},
        
        # Returning to home screen
        {"source": "learn_competencies", "dest": "home_screen", "trigger": "next"},
        # {"source": "assessment", "dest": "home_screen", "trigger": "next"},
        # {"source": "resources_tips", "dest": "home_screen", "trigger": "next"},
        # {"source": "help_support", "dest": "home_screen", "trigger": "next"},
        
        # End state
        {"source": "thankyou_message", "dest": "home_screen", "trigger": "next"},
    ]

    conditions = {}
    output_variables = set()
    variable_names = TeacherCompetencyVariables

    def is_learn_competencies_selected(self):
        return self.variables.home_screen_option == "competencies"
    
    def is_assessment_selected(self):
        return self.variables.home_screen_option == "assessment"
    
    def is_resources_tips_selected(self):
        return self.variables.home_screen_option == "resources"
    
    def is_help_support_selected(self):
        return self.variables.home_screen_option == "help"

    def is_send_another_question(self):
        if self.variables.total_num_question == self.variables.question_count:
            return False
        return True

    def __init__(self, send_message: callable, credentials: Dict[str, Any] = None):

        if credentials is None:
            credentials = {}

        self.credentials = {}

        self.credentials["AZURE_OPENAI_API_KEY"] = credentials.get(
            "AZURE_OPENAI_API_KEY"
        )
        if not self.credentials["AZURE_OPENAI_API_KEY"]:
            raise ValueError("Missing credential: AZURE_OPENAI_API_KEY")
        self.credentials["AZURE_OPENAI_API_VERSION"] = credentials.get(
            "AZURE_OPENAI_API_VERSION"
        )
        if not self.credentials["AZURE_OPENAI_API_VERSION"]:
            raise ValueError("Missing credential: AZURE_OPENAI_API_VERSION")
        self.credentials["AZURE_OPENAI_API_ENDPOINT"] = credentials.get(
            "AZURE_OPENAI_API_ENDPOINT"
        )
        if not self.credentials["AZURE_OPENAI_API_ENDPOINT"]:
            raise ValueError("Missing credential: AZURE_OPENAI_API_ENDPOINT")

        if not credentials.get("FAST_MODEL"):
            raise ValueError("Missing credential: FAST_MODEL")
        self.credentials["FAST_MODEL"] = credentials.get("FAST_MODEL")

        if not credentials.get("SLOW_MODEL"):
            raise ValueError("Missing credential: SLOW_MODEL")
        self.credentials["SLOW_MODEL"] = credentials.get("SLOW_MODEL")

        self.plugins: Dict[str, AbstractFSM] = {}
        self.variables = self.variable_names()
        super().__init__(send_message=send_message)

    def standard_ask_again(self, message=None):
        self.status = Status.WAIT_FOR_ME
        if message is None:
            message = "Sorry, I did not understand your question. Can you tell me again?"
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def on_enter_language_selection(self):
        self._on_enter_select_language()

    def on_enter_welcome_message_display(self):
        self.status = Status.WAIT_FOR_ME
        message = "Welcome to the Teacher Competency Tool! Let's start your journey to becoming an even more amazing teacher! 😊"
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def on_enter_collect_name(self):
        self.status = Status.WAIT_FOR_ME
        message = "Please enter your name."
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_collect_name_logic(self):
        self.status = Status.WAIT_FOR_ME
        setattr(self.variables, "name", self.current_input)
        self.status = Status.MOVE_FORWARD

    def on_enter_collect_subject(self):
        self.status = Status.WAIT_FOR_ME
        message = "What subject do you teach?"
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_collect_subject_logic(self):
        self.status = Status.WAIT_FOR_ME
        setattr(self.variables, "subject", self.current_input)
        self.status = Status.MOVE_FORWARD

    def on_enter_collect_grade(self):
        self.status = Status.WAIT_FOR_ME
        message = "Which grade do you teach?"
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_collect_grade_logic(self):
        self.status = Status.WAIT_FOR_ME
        setattr(self.variables, "grade", self.current_input)
        self.status = Status.MOVE_FORWARD

    def on_enter_confirmation(self):
        self.status = Status.WAIT_FOR_ME
        message = f"Thank you, {self.variables.name}! You teach {self.variables.subject} for Grade {self.variables.grade}. Let's begin!"
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def on_enter_home_screen(self):
        self.status = Status.WAIT_FOR_ME
        message = "What would you like to do today?"
        slots = [
            Option(option_id="competencies", option_text="Learn about competencies"),
            Option(option_id="assessment", option_text="Take an assessment"),
            Option(option_id="resources", option_text="Resources & Tips"),
            Option(option_id="help", option_text="Help/Support"),
        ]
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.BUTTON,
                    button=ButtonMessage(
                        body=message,
                        header="",
                        footer="",
                        options=slots,
                    ),
                ),
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_home_screen_logic(self):
        self.status = Status.WAIT_FOR_ME
        options_list = json.loads(self.current_input)
        option_id = options_list[0]["option_id"]
        setattr(self.variables, "home_screen_option", option_id)
        self.status = Status.MOVE_FORWARD

    def on_enter_learn_competencies(self):
        self.status = Status.WAIT_FOR_ME
        grade = self.variables.grade
        subject = self.variables.subject
        message = f"Let's learn about the essential competencies for teaching grade *{grade}* *{subject}*.\n"
        prompt = f"""
            The user is teaching grade {grade} and has selected '{subject}' as their subject.  
            Provide a list of **essential teaching competencies** that are most relevant for a teacher  
            instructing grade {grade} students in {subject}.  

            The competencies should be **practical, competency-based, and aligned with best teaching practices**.  
            They should cover the following key areas:
            1. **Pedagogical Approaches** - Effective teaching methods for {subject} at grade {grade}.  
            2. **Subject Mastery** - Key topics and depth of knowledge required.  
            3. **Assessment Techniques** - Best practices for evaluating student progress.  
            4. **Student Engagement** - Strategies to keep students motivated and active in learning.  
            5. **Technology Integration** - Useful digital tools and techniques to enhance learning.  

            Ensure the response is structured in **valid JSON format** as follows:

            ```json
            {{
                "competencies": [
                    {{
                        "category": "Category Name",
                        "description": "Brief explanation of the competency",
                        "examples": ["Example 1", "Example 2"]
                    }},
                    ...
                ]
            }}
            ```

            Only return **the JSON output** without any extra text. Also, return only 2 competencies.
        """

        result = Parser.parse_user_input(
            prompt,
            options=None,
            user_input=None,
            azure_endpoint=self.credentials["AZURE_OPENAI_API_ENDPOINT"],
            azure_openai_api_key=self.credentials["AZURE_OPENAI_API_KEY"],
            azure_openai_api_version=self.credentials["AZURE_OPENAI_API_VERSION"],
            model="gpt-4",
        )
        for competency in result['competencies']:
            message += f"*Category:* {competency['category']}\n*Description:* {competency['description']}\n*Examples:* {', '.join(competency['examples'])}"
            self.send_message(
                FSMOutput(
                    intent=FSMIntent.SEND_MESSAGE,
                    message=Message(
                        message_type=MessageType.TEXT, text=TextMessage(body=message)
                    ),
                )
            )
            message = ""
        self.status = Status.MOVE_FORWARD

    def on_enter_assessment(self):
        self.status = Status.WAIT_FOR_ME
        setattr(self.variables, "question_count", 0)
        setattr(self.variables, "user_input_param", [])
        setattr(self.variables, "question_list", None)
        setattr(self.variables, "current_question_id", None)
        message = f"Ready for a quick quiz on {self.variables.subject}? Let's go! 😊"
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def on_enter_preparing_question(self):
        self.status = Status.WAIT_FOR_ME
        message = "Please wait while we are preparing question for you."
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def get_question_from_openai(self):
            
        grade = self.variables.grade
        subject = self.variables.subject
        total_num_question = self.variables.total_num_question

        task = f"""
        User is teaching a foundation for grade {grade}, and they selected '{subject}' as their subject.  

        ### **Generate a Fresh, Randomized Quiz**
        - First, generate **at least 5x the required number of unique questions**.  
        - Then, **randomly select** {total_num_question} questions from this pool.  
        - Ensure **no two quiz attempts contain the same set of questions**.  
        - Each question must be **different in every quiz attempt**—no repetition from previous runs.  
        - Each question must have a **unique question_id**.  
        - Questions should be **clear, competency-based, and relevant** to the subject.  

        ### **Strict JSON Output Format:**
        {{
            "result": [
                {{
                    "question_id": 1,
                    "question": "Sample question?",
                    "options": {{
                        "A": "Option 1",
                        "B": "Option 2",
                        "C": "Option 3",
                        "D": "Option 4"
                    }},
                    "answer": "Correct answer"
                }}
            ]
        }}
        """

        result = Parser.parse_user_input(
            task,
            options=None,
            user_input=None,
            azure_endpoint=self.credentials["AZURE_OPENAI_API_ENDPOINT"],
            azure_openai_api_key=self.credentials["AZURE_OPENAI_API_KEY"],
            azure_openai_api_version=self.credentials["AZURE_OPENAI_API_VERSION"],
            model="gpt-4",
        )
        result = result['result']
        try:
            setattr(self.variables, "question_list", result)
        except Exception as e:
            setattr(self.variables, "question_list", None)

        return result

    def on_enter_show_question(self):
        self.status = Status.WAIT_FOR_CALLBACK
        
        question = ""
        question_id = ""

        if self.variables.question_list is None:
            get_question = self.get_question_from_openai()
            question = 'Question 1: ' + get_question[0]["question"]
            question_id = get_question[0]["question_id"]
            options = get_question[0]["options"]

        if self.variables.question_count != 0:
            next_question_num = self.variables.question_count
            question = f'Question {next_question_num + 1}: ' + self.variables.question_list[next_question_num]["question"]
            question_id = self.variables.question_list[next_question_num]["question_id"]
            options = self.variables.question_list[next_question_num]["options"]

        # logger.error("new question id and question {} {} {}".format(question_id, question, type(question)))
        options_list = []
        for key, value in options.items():
            options_list.append(Option(option_id=key, option_text=value))

        message = question
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.OPTION_LIST,
                    option_list=ListMessage(
                        body=message,
                        header="",
                        footer="",
                        button_text="Question Answer",
                        list_title="Question Answer",
                        options=options_list,
                    ),
                ),
            )
        )
        try:
            # update the pydantic variable with the result
            setattr(self.variables, "current_question_id", question_id)
        except Exception as e:
            setattr(self.variables, "current_question_id", None)

        self.status = Status.MOVE_FORWARD

    def on_enter_question_answer_input(self):
        self.status = Status.WAIT_FOR_ME
        value = self.variables.question_count
        setattr(self.variables, "question_count", value + 1)
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_answer_input_logic(self):
        self.status = Status.WAIT_FOR_ME

        current_question_id = self.variables.current_question_id
        question_list = self.variables.question_list

        current_ques_param = [ques_list for ques_list in question_list if ques_list["question_id"] == current_question_id]
        curret_user_input_param = {
            "question_id": current_question_id,
            "question": current_ques_param[0]["question"],
            "options": current_ques_param[0]["options"],
            "answer": current_ques_param[0]["answer"],
            "selected_option": self.current_input,
        }
        updated_user_input = self.variables.user_input_param
        updated_user_input.append(curret_user_input_param)
        try:
            # update the pydantic variable with the result
            setattr(self.variables, "user_input_param", updated_user_input)
        except Exception as e:
            setattr(self.variables, "user_input_param", self.variables.user_input_param)
        
        self.status = Status.MOVE_FORWARD

    def on_enter_waiting_message(self):
        self.status = Status.WAIT_FOR_ME
        message = "Please wait while we are calculating your score"

        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def on_enter_calculate_score(self):
        self.status = Status.WAIT_FOR_ME
        
        task = f"""User is teaching for grade {self.variables.grade} of subject '{self.variables.subject}',
            these are the objective type question '{self.variables.question_list}'.
            User provides the list of question with selected option such as (A/B/C/D) as 'selected_option'.
            Calculate the score of each competency and show the output such as Your score is (5/10)'.
            Return the output in json format: {{'result': <input>}}."""
        
        user_input = f" user input is {self.variables.user_input_param}"
        result = Parser.parse_user_input(
            task,
            options=None,
            user_input=user_input,
            azure_endpoint=self.credentials["AZURE_OPENAI_API_ENDPOINT"],
            azure_openai_api_key=self.credentials["AZURE_OPENAI_API_KEY"],
            azure_openai_api_version=self.credentials["AZURE_OPENAI_API_VERSION"],
            model="gpt-4",
        )

        message = result['result']
        # sequence = 1
        # for competency, score in result.items():
        #     message += f"{sequence}. {competency}    {score} \n \n"
        #     sequence += 1

        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def on_enter_thankyou_message(self):
        self.status = Status.WAIT_FOR_ME
        setattr(self.variables, "question_count", 0)
        setattr(self.variables, "user_input_param", [])
        setattr(self.variables, "question_list", None)
        setattr(self.variables, "current_question_id", None)
        message = "Thank you for attempting MX Quiz!"
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def on_enter_resources_tips(self):
        self.status = Status.WAIT_FOR_ME
        message = "Here are some resources and tips for you.\n"
        grade = self.variables.grade
        subject = self.variables.subject

        prompt = f"""
            The user is teaching grade {grade} and has selected '{subject}' as their subject.  
            Provide a list of **recommended reading resources** (books, articles, websites, and PDFs)  
            that are best suited for grade {grade} students learning {subject}.
            The resources should be **competency-based** and **relevant** to the subject.
            The recources should be **clear, concise, and easy to understand**.
            The resources should be **free** or open to public.  
            Ensure the response is structured in **valid JSON format** as follows:

            ```json
            {{
                "resources": [
                    {{
                        "title": "Book/Article Name",
                        "author": "Author Name",
                        "description": "Brief summary of the resource",
                        "link": "URL of the resource"
                    }},
                    ...
                ]
            }}
            ```

            Only return **the JSON output** without any extra text also return only 2 resouces.
        """
        result = Parser.parse_user_input(
            prompt,
            options=None,
            user_input=None,
            azure_endpoint=self.credentials["AZURE_OPENAI_API_ENDPOINT"],
            azure_openai_api_key=self.credentials["AZURE_OPENAI_API_KEY"],
            azure_openai_api_version=self.credentials["AZURE_OPENAI_API_VERSION"],
            model="gpt-4",
        )
        for resource in result['resources']:
            message += f"**Title:** {resource['title']}\n**Author:** {resource['author']}\n**Description:** {resource['description']}\n**Link:** {resource['link']}"
            self.send_message(
                FSMOutput(
                    intent=FSMIntent.SEND_MESSAGE,
                    message=Message(
                        message_type=MessageType.TEXT, text=TextMessage(body=message)
                    ),
                )
            )
            message = ""
        self.status = Status.MOVE_FORWARD

    def on_enter_help_support(self):
        self.status = Status.WAIT_FOR_ME
        message = "How can we assist you today? Please describe your issue."
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_end(self):
        self.status = Status.WAIT_FOR_ME
        message = "Thank you for using the MX Teacher Competency Tool. Have a great day!"
        self.send_message(
            FSMOutput(
                intent=FSMIntent.SEND_MESSAGE,
                message=Message(
                    message_type=MessageType.TEXT, text=TextMessage(body=message)
                ),
            )
        )
        self.status = Status.MOVE_FORWARD