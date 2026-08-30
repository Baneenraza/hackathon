"""Generate the 3 synthetic knowledge-base PDFs for RAG.

These stand in for real plant documentation (not supplied with the dataset).
Content is written to be internally consistent with the AI4I 2020 schema:
machine Types L/M/H and failure modes TWF (tool wear), HDF (heat dissipation),
PWF (power), OSF (overstrain), RNF (random). Numbered sections so the RAG
chunker can split cleanly and cite "<doc> section <n.n>".
"""
import os
from fpdf import FPDF

OUT = os.path.join("data", "knowledge_base")
os.makedirs(OUT, exist_ok=True)


class Doc(FPDF):
    def __init__(self, title):
        super().__init__()
        self.doc_title = title
        self.set_auto_page_break(auto=True, margin=18)
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.multi_cell(0, 9, title)
        self.ln(2)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(90, 90, 90)
        self.multi_cell(0, 5, "Synthetic reference document - AI Factory Intelligence "
                              "Command Center. For demonstration only; not a real "
                              "operating procedure.")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def section(self, num, heading, body):
        self.set_font("Helvetica", "B", 12)
        self.multi_cell(0, 7, f"{num}  {heading}")
        self.ln(1)
        self.set_font("Helvetica", "", 10.5)
        for para in body.strip().split("\n\n"):
            self.multi_cell(0, 5.6, para.strip())
            self.ln(1.5)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"{self.doc_title}  -  page {self.page_no()}", align="C")


def build(title, filename, sections):
    d = Doc(title)
    for num, heading, body in sections:
        d.section(num, heading, body)
    path = os.path.join(OUT, filename)
    d.output(path)
    print("wrote", path)


# ----------------------------------------------------------------------------
safety_sop = [
    ("1.0", "Purpose and Scope",
     "This Standard Operating Procedure defines the mandatory safety steps for "
     "operators and maintenance technicians working on CNC milling and turning "
     "cells classified as Type L (low-grade), Type M (medium-grade) and Type H "
     "(high-grade). It applies to all planned maintenance, unplanned "
     "intervention, and inspection activities.\n\n"
     "No predictive alert from the Command Center authorises autonomous action. "
     "Every intervention requires a named human approver as described in "
     "section 6.0."),
    ("2.0", "Lockout / Tagout (LOTO)",
     "2.1 Before any physical work, isolate the machine at the main disconnect, "
     "apply a personal lock, and verify zero energy state at the spindle drive "
     "and hydraulic accumulator.\n\n"
     "2.2 Rotating spindles must come to a complete mechanical stop. Do not rely "
     "on the controller e-stop as an energy isolation device.\n\n"
     "2.3 Tag must record technician name, work order number, and expected "
     "restoration time."),
    ("3.0", "Response to Overstrain (OSF) Conditions",
     "3.1 An overstrain condition is indicated when the product of tool wear and "
     "torque exceeds the Type-specific threshold (Type L approx 11,000 "
     "min*Nm, Type M approx 12,000, Type H approx 13,000).\n\n"
     "3.2 On an OSF alert, immediately reduce feed rate and bring the axis load "
     "below 50 percent. Do not continue the current load-bearing pass.\n\n"
     "3.3 Inspect the tool for chipping and the workpiece for deformation before "
     "restarting. Replace the tool if wear is above 200 minutes."),
    ("4.0", "Response to Heat Dissipation (HDF) Conditions",
     "4.1 HDF risk rises when the air-to-process temperature difference falls "
     "below 8.6 K while rotational speed is under 1380 rpm.\n\n"
     "4.2 Confirm coolant flow and check the heat exchanger inlet filter. "
     "Clear any chip blockage around the spindle housing.\n\n"
     "4.3 If the temperature difference cannot be restored within 10 minutes, "
     "stop the machine and raise a maintenance work order at HIGH urgency."),
    ("5.0", "Response to Power (PWF) and Tool Wear (TWF) Conditions",
     "5.1 Power failure risk (PWF) occurs when the mechanical power (torque "
     "times angular velocity) leaves the 3.5 - 9 kW band. Check drive tuning and "
     "supply voltage; a sustained low-power reading often precedes a stall.\n\n"
     "5.2 Tool wear failure (TWF) is expected between 200 and 240 minutes of "
     "cumulative tool wear. Schedule a tool change at 200 minutes; never run a "
     "tool past 240 minutes.\n\n"
     "5.3 Random failures (RNF) have no measurable precursor. Treat any RNF flag "
     "as advisory and corroborate with vibration and acoustic inspection."),
    ("6.0", "Human Approval and Escalation",
     "6.1 The Command Center issues DECISION SUPPORT only. A maintenance "
     "supervisor must APPROVE, REJECT or MODIFY every recommendation and record "
     "the reason.\n\n"
     "6.2 HIGH risk recommendations require sign-off by the shift engineer in "
     "addition to the supervisor.\n\n"
     "6.3 If a recommendation is rejected, the machine may only keep running "
     "under increased monitoring with the residual risk logged in the incident "
     "report."),
    ("7.0", "Personal Protective Equipment",
     "Safety glasses and cut-resistant gloves are mandatory in all machining "
     "cells. Hearing protection is required when rotational speed exceeds "
     "2000 rpm. Gloves must not be worn while a spindle is rotating."),
]

maintenance_manual = [
    ("1.0", "Machine Types and Duty Ratings",
     "Type L cells are rated for light-duty batch work and tolerate the widest "
     "process variation. Type M cells are the general-purpose workhorse. Type H "
     "cells run tight-tolerance parts and have the lowest allowable overstrain "
     "margin. Maintenance intervals scale with duty: L every 500 machine-hours, "
     "M every 350, H every 250."),
    ("2.0", "Preventive Maintenance Schedule",
     "2.1 Daily: check coolant level and concentration, inspect way covers, "
     "confirm air temperature is within 295 - 305 K and process temperature is "
     "no more than 12 K above air temperature.\n\n"
     "2.2 Weekly: lubricate ball screws, verify spindle runout, download the "
     "controller fault log.\n\n"
     "2.3 At each PM interval: replace spindle coolant filter, check drive belt "
     "tension, calibrate the torque sensor per the Calibration Checklist."),
    ("3.0", "Tool Wear Management (TWF)",
     "3.1 Tool wear is tracked in cumulative minutes of cutting time. The wear "
     "life for standard carbide tooling is 200 minutes nominal, 240 minutes "
     "absolute maximum.\n\n"
     "3.2 Log every tool change with the tool ID and the wear reading at "
     "removal. A tool removed below 150 minutes indicates a feeds-and-speeds "
     "problem to investigate.\n\n"
     "3.3 Worn tools raise cutting torque, which compounds overstrain risk on "
     "Type H cells. Prioritise tool changes there."),
    ("4.0", "Thermal System and Heat Dissipation (HDF)",
     "4.1 The chiller must hold the process-to-air temperature difference at or "
     "above 8.6 K under load. A shrinking difference with rotational speed below "
     "1380 rpm is the classic HDF precursor.\n\n"
     "4.2 Clean the heat exchanger monthly. A fouled exchanger is the most "
     "common root cause of repeat HDF alerts.\n\n"
     "4.3 If HDF alerts repeat after cleaning, check the coolant pump impeller "
     "and the thermostatic expansion valve."),
    ("5.0", "Drivetrain, Power and Overstrain (PWF, OSF)",
     "5.1 Mechanical power should sit between 3.5 and 9 kW during a normal cut. "
     "Readings outside that band trip the PWF logic. Low power with rising "
     "torque means the spindle is bogging down - reduce depth of cut.\n\n"
     "5.2 Overstrain (OSF) is driven by tool wear times torque. Keep the "
     "product under the Type threshold (L 11000, M 12000, H 13000 min*Nm).\n\n"
     "5.3 After any OSF event, inspect the spindle bearings and the axis "
     "couplings for backlash before returning the machine to service."),
    ("6.0", "Corrective Maintenance Workflow",
     "6.1 Triage the alert against sensor history in the Command Center. "
     "Confirm the failure mode (TWF/HDF/PWF/OSF/RNF).\n\n"
     "6.2 Raise a work order with urgency LOW, MEDIUM or HIGH. HIGH is reserved "
     "for conditions that can cause tool breakage, thermal damage or an "
     "unplanned stop within one shift.\n\n"
     "6.3 Perform LOTO per the Safety SOP section 2.0. Complete the repair, "
     "update the maintenance log, and run a 15-minute verification cut before "
     "release."),
    ("7.0", "Spare Parts and Consumables",
     "Keep on hand: carbide insert sets for each cell, two spindle coolant "
     "filters per machine, one spare drive belt per Type, and a calibrated "
     "reference torque cell. Reorder when stock falls below 30 percent."),
    ("8.0", "Record Keeping",
     "Every intervention must produce a dated record containing the machine ID, "
     "failure mode, parts used, technician, approver, and the post-repair "
     "verification result. Records feed the reliability model and must not be "
     "skipped."),
]

calibration_checklist = [
    ("1.0", "Scope",
     "This checklist covers periodic calibration of the sensors that feed the "
     "predictive models: air and process temperature, rotational speed, torque, "
     "and the tool-wear timer. Perform at every preventive maintenance interval "
     "and after any sensor replacement."),
    ("2.0", "Pre-Calibration Checks",
     "2.1 Machine at ambient, powered on for at least 30 minutes for thermal "
     "stability.\n\n"
     "2.2 Reference instruments in date: calibrated RTD, optical tachometer, "
     "reference torque cell.\n\n"
     "2.3 Record ambient air temperature; it should read 295 - 305 K."),
    ("3.0", "Temperature Sensor Calibration",
     "3.1 Compare the air-temperature RTD against the reference at two points. "
     "Acceptance: within 0.5 K.\n\n"
     "3.2 Compare process-temperature RTD similarly. Acceptance: within 0.5 K.\n\n"
     "3.3 Verify the derived air-to-process difference reads at least 8.6 K "
     "under a light test load. A smaller reading with low rpm is the HDF "
     "warning band - investigate before returning to service."),
    ("4.0", "Rotational Speed Calibration",
     "4.1 Command 1500 rpm and measure with the optical tachometer. "
     "Acceptance: within 15 rpm.\n\n"
     "4.2 Repeat at 2500 rpm. Acceptance: within 25 rpm.\n\n"
     "4.3 Confirm the controller flags speeds below 1380 rpm as an HDF-relevant "
     "operating region."),
    ("5.0", "Torque Sensor Calibration",
     "5.1 Apply known loads with the reference torque cell at 10, 30 and 50 Nm. "
     "Acceptance: within 1 Nm or 2 percent, whichever is larger.\n\n"
     "5.2 Check that the overstrain calculation (tool wear times torque) uses "
     "the corrected torque value.\n\n"
     "5.3 Verify the Type-specific OSF thresholds are configured: L 11000, "
     "M 12000, H 13000 min*Nm."),
    ("6.0", "Tool-Wear Timer Verification",
     "6.1 Confirm the cumulative wear counter increments only during spindle-on "
     "cutting time.\n\n"
     "6.2 Verify the 200-minute change reminder and the 240-minute hard stop "
     "are both active.\n\n"
     "6.3 Reset the counter only when a tool change is logged with a tool ID."),
    ("7.0", "Power Calculation Check",
     "Confirm mechanical power is computed as torque times angular velocity and "
     "that the PWF band of 3.5 - 9 kW is configured. Simulate an out-of-band "
     "value and confirm the alert fires."),
    ("8.0", "Sign-Off",
     "Record every measured value, pass/fail, the technician, and the approving "
     "supervisor. Any FAIL requires a repeat calibration after correction. "
     "File the completed checklist with the machine maintenance record."),
]

build("Factory Safety Standard Operating Procedure", "safety_sop.pdf", safety_sop)
build("CNC Cell Maintenance Manual", "maintenance_manual.pdf", maintenance_manual)
build("Sensor Calibration Checklist", "calibration_checklist.pdf", calibration_checklist)
print("done")
