"""Build the checked-in tender compliance paraphrase benchmark.

Run from the repository root. Each independently verified tender fact has ten
realistic compliance-matrix query formulations so retrieval can be tested for
robustness to analyst wording, not just memorized keywords.
"""
from __future__ import annotations

import json
from pathlib import Path


INTENTS = [
    {
        "intent": "gpu_warranty",
        "queries": [
            "What warranty is mandatory for the GPU server?",
            "State the minimum GPU server warranty period and coverage.",
            "Does the GPU server require onsite warranty support from the OEM?",
            "Extract the warranty compliance requirement for parts, labour, and service calls.",
            "Who must attend onsite service calls during the server warranty?",
            "Prepare the compliance-matrix entry for GPU server warranty.",
            "Is a three-year comprehensive warranty required for the proposed server?",
            "Identify the minimum acceptable warranty and responsible service engineer.",
            "What must a bidder confirm against the warranty clause?",
            "Summarize the server warranty obligation for technical compliance review."
        ],
        "model_answer": "Minimum three-year comprehensive parts-and-labour warranty; onsite service calls must be attended directly by an OEM engineer.",
        "answer_evidence": [["minimum 3 years", "minimum three years"], "onsite service calls", "OEM engineer"],
        "relevant": [{"source": "hardware_networking_05_gpu_server.pdf", "pages": [5]}]
    },
    {
        "intent": "gpu_security",
        "queries": [
            "List the mandatory security controls for the GPU server.",
            "Does the proposed server comply with secure boot, TPM 2.0, and secure erase requirements?",
            "Extract the hardware root-of-trust requirement from the technical specification.",
            "What firmware and BIOS security features must the GPU server support?",
            "Prepare a compliance entry for server platform security.",
            "Is encryption of data at rest mandatory for the GPU server?",
            "Which security capabilities must be available from day one?",
            "Summarize the GPU server cyber-security requirements for bid evaluation.",
            "What must the bidder offer for secure boot, breach protection, and key management?",
            "Identify all security specification points requiring OEM compliance."
        ],
        "model_answer": "The server requires secure boot, hardware or dual root of trust, secure firmware and BIOS recovery, breach protection, TPM 2.0, instant secure erase, and encryption of data at rest.",
        "answer_evidence": ["secure boot", "root of trust", "TPM 2 0", "instant secure erase", "data at rest"],
        "relevant": [{"source": "hardware_networking_05_gpu_server.pdf", "pages": [5]}]
    },
    {
        "intent": "gpu_local_support",
        "queries": [
            "What local support facility must the GPU server OEM maintain?",
            "Is an OEM service centre in Delhi a mandatory condition?",
            "Extract the documentary proof required for the Delhi support centre.",
            "Prepare the local-support row of the GPU server compliance matrix.",
            "Must the OEM list its Delhi service and support centre online?",
            "What evidence establishes compliance with the local support requirement?",
            "Summarize the OEM presence required in Delhi.",
            "Does a bidder comply if its OEM lacks a registered Delhi service centre?",
            "Identify the website and government-document requirements for local support.",
            "State every condition attached to the server OEM's Delhi support centre."
        ],
        "model_answer": "The OEM must have its own registered service and support centre in Delhi, list it on the OEM website, and provide documentary evidence issued by a government department.",
        "answer_evidence": ["registered service and support center in Delhi", "OEM website", "documentary evidence issued by Govt department"],
        "relevant": [{"source": "hardware_networking_05_gpu_server.pdf", "pages": [5]}]
    },
    {
        "intent": "gpu_installation",
        "queries": [
            "Who is responsible for installing and commissioning the GPU server solution?",
            "Must OEM engineers perform installation, testing, and training?",
            "Are implementation costs required to be included from day one?",
            "Extract the GPU server installation compliance clause.",
            "Prepare a compliance-matrix entry for installation and training responsibility.",
            "What activities must be performed by OEM engineers?",
            "Does the quoted solution need to include testing and implementation costs?",
            "Summarize the day-one deployment obligation for the GPU server.",
            "Identify who must carry out installation, testing, training, and implementation.",
            "What should the technical bid confirm about OEM-led deployment?"
        ],
        "model_answer": "Installation, testing, training, and implementation must be done by OEM engineers, and all associated costs must be included from day one.",
        "answer_evidence": ["installation testing training and implementation", "included from day one", "OEM Engineers"],
        "relevant": [{"source": "hardware_networking_05_gpu_server.pdf", "pages": [5]}]
    },
    {
        "intent": "gpu_debarment",
        "queries": [
            "What debarment declaration is required from the GPU server OEM and bidder?",
            "Can an OEM banned for more than three months in the last five years qualify?",
            "Extract the holiday-period and debarment eligibility clause.",
            "Prepare the bidder/OEM debarment row for the compliance matrix.",
            "What historical period applies to the government-organization ban check?",
            "State the disqualification rule concerning bans and debarment.",
            "Does the compliance condition apply to both the OEM and bidder?",
            "Summarize the three-month debarment restriction.",
            "What must be verified about the OEM's status during the last five years?",
            "Identify the mandatory non-debarment condition for this server procurement."
        ],
        "model_answer": "The OEM and bidder must not have been put on holiday, banned, or debarred by a government organization for more than three months during the last five years.",
        "answer_evidence": [["banned or debarred", "holiday period or banned"], [">3 months", "3 months"], "last 5 years"],
        "relevant": [{"source": "hardware_networking_05_gpu_server.pdf", "pages": [5]}]
    },
    {
        "intent": "nhb_contract_term",
        "queries": [
            "What is the total term of the NHB SAP support contract?",
            "How long is the initial NHB work order and when may it be renewed?",
            "Extract the contract-duration requirement for SAP ERP support.",
            "Prepare the contract-period row for the NHB compliance matrix.",
            "Is renewal of the SAP support work order performance dependent?",
            "State the initial and maximum service periods under the NHB RFP.",
            "What satisfactory-performance condition applies to contract renewal?",
            "Summarize the five-year SAP support engagement structure.",
            "Does NHB award all five years at once or begin with one year?",
            "Identify the duration and renewal terms a bidder must accept."
        ],
        "model_answer": "The service contract is for five years. The work order is initially for one year and may be renewed following a satisfactory performance review.",
        "answer_evidence": [["5 years", "five years"], ["initially placed for 1 year", "initial work order is for one year"], "satisfactory performance"],
        "relevant": [{"source": "it_software_02_sap_erp_support.pdf", "pages": [9]}]
    },
    {
        "intent": "nhb_fees_emd",
        "queries": [
            "What are the tender fee and EMD for the NHB SAP RFP?",
            "Is the Rs. 5,000 RFP fee refundable?",
            "Is the Rs. 200,000 earnest money deposit refundable?",
            "Extract the bid cost and security amounts for the commercial compliance matrix.",
            "Prepare the RFP-fee and EMD row for the NHB tender.",
            "How much must a bidder pay for the RFP and deposit as earnest money?",
            "State which NHB bid payment is refundable and which is non-refundable.",
            "Summarize the upfront tender payments required from an NHB bidder.",
            "What monetary conditions appear in the NHB bid summary?",
            "Identify the exact RFP cost, EMD amount, and refund status."
        ],
        "model_answer": "The RFP costs Rs. 5,000 and is non-refundable; the EMD is Rs. 200,000 and is refundable.",
        "answer_evidence": ["Rs 5 000", "non refundable", "Rs 200 000", "refundable"],
        "relevant": [{"source": "it_software_02_sap_erp_support.pdf", "pages": [2]}]
    },
    {
        "intent": "nhb_prebid_queries",
        "queries": [
            "What is the deadline for submitting NHB pre-bid clarification questions?",
            "How may bidders send written queries for the SAP RFP?",
            "Are clarification requests accepted after the pre-bid meeting?",
            "Extract the pre-bid query process for the compliance matrix.",
            "Prepare the clarification deadline and submission-method row.",
            "Must NHB receive bidder questions by email or post before 24 July 2017?",
            "State the rule governing late pre-bid queries.",
            "Summarize how and when an NHB bidder should raise RFP doubts.",
            "What bidder action is required before the SAP pre-bid meeting?",
            "Identify the cutoff date and permitted channels for clarification queries."
        ],
        "model_answer": "Written queries must reach NHB by email or post on or before 24 July 2017; queries received after the pre-bid meeting are not entertained.",
        "answer_evidence": [["24 07 2017", "24 July 2017"], "e mail or by post", "after the pre bid meeting"],
        "relevant": [{"source": "it_software_02_sap_erp_support.pdf", "pages": [12]}]
    },
    {
        "intent": "housekeeping_eligibility",
        "queries": [
            "What turnover threshold applies to the Subarnapur housekeeping bidder?",
            "Are consortium bids allowed for the hospital housekeeping contract?",
            "Which financial years are used to assess average annual turnover?",
            "Extract the turnover and consortium rules for the eligibility matrix.",
            "Prepare the financial-capacity row for the housekeeping tender.",
            "Must a bidder average at least Rs. 3 crore over the specified three years?",
            "State the legal teaming restriction and turnover qualification.",
            "Summarize the consortium and revenue eligibility criteria.",
            "Does the tender permit a consortium to meet the turnover requirement?",
            "Identify the minimum turnover and relevant fiscal years for qualification."
        ],
        "model_answer": "Consortiums are not allowed. Average annual turnover must be at least Rs. 3 crore during 2022-23, 2023-24, and 2024-25.",
        "answer_evidence": ["consortium is not allowed", ["Rs 3 Crores", "3 crore"], "2022 23", "2023 24", "2024 25"],
        "relevant": [{"source": "services_consulting_01_hospital_housekeeping.pdf", "pages": [4]}]
    },
    {
        "intent": "ups_tender_summary",
        "queries": [
            "What is the scope and estimated value of the IISc IDR UPS tender?",
            "How many 300 KVA online UPS units are required?",
            "What completion period applies to the UPS supply and commissioning work?",
            "How much earnest money must be deposited for the IISc UPS tender?",
            "Prepare the scope, value, duration, and EMD rows for the compliance matrix.",
            "Extract the key commercial facts from the IISc UPS tender notification.",
            "Does the work cover four UPS units with battery backup for the IDR building?",
            "State the estimated contract value and work-completion deadline.",
            "Summarize the tender requirement for 300 KVA online UPS equipment.",
            "Identify the quantity, capacity, estimated value, duration, and bid security."
        ],
        "model_answer": "Supply, installation, testing, and commissioning of four 300 KVA online UPS units with battery backup; estimated value Rs. 2.7 crore plus GST, completion in four months, and EMD Rs. 4,05,000.",
        "answer_evidence": [["4 Nos 300 KVA", "four 300 KVA"], "Rs 2 7 crores GST", ["4 Months", "four months"], "Rs 4 05 000"],
        "relevant": [{"source": "electrical_mechanical_05_ups_system.pdf", "pages": [3]}]
    }
]


def build_cases():
    cases = []
    for intent in INTENTS:
        for number, query in enumerate(intent["queries"], 1):
            cases.append({"case_id": f'{intent["intent"]}_{number:02d}', "intent": intent["intent"],
                          "query": query, "model_answer": intent["model_answer"],
                          "answer_evidence": intent["answer_evidence"], "relevant": intent["relevant"]})
    return cases


if __name__ == "__main__":
    output = Path(__file__).with_name("tender_compliance_cases.jsonl")
    output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in build_cases()), encoding="utf-8")
