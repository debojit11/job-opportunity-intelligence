from app.services.assessment import (assess_job_opportunity,)

from app.presentation.console import (print_assessment_report,)


def main():
    assessment = assess_job_opportunity()

    print_assessment_report(assessment)


if __name__ == "__main__":
    main()