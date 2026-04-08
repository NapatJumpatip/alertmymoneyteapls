# FinScan — Personal Finance Anomaly Detector

> วิเคราะห์รายจ่ายผิดปกติจาก CSV ธนาคาร ด้วย Isolation Forest + IQR · รันใน browser ไม่ต้องติดตั้งอะไร

## Demo

อัปโหลดไฟล์ CSV จาก KBank แล้วดูผลได้ทันที

## Features

| Tab | ฟีเจอร์ |
|---|---|
| Overview | Anomaly timeline, category donut, insight + คำแนะนำ |
| Categories | Filter แยกหมวด, bar chart ยอดต่อหมวด |
| Trends | Monthly comparison, Calendar heatmap รายวัน |
| Ranking | Top 10 รายการ, Top anomaly ranking |
| Report | สรุปทั้งหมด + Export PDF |

## Algorithm

```
CSV Input
   ↓
Feature Engineering
  - Rolling mean 7 วัน
  - Z-score เทียบ window
  - Day of week, Month
   ↓
Isolation Forest (100 trees, JS port)
  + IQR Rule (Q3 + 1.5×IQR)
   ↓
Anomaly Score → Flag + Reason
   ↓
Interactive Dashboard
```

**ทำไม Isolation Forest?**  
เป็น unsupervised model — ไม่ต้อง label ข้อมูลว่าอันไหนผิดปกติก่อน  
ตัว model สร้าง random decision trees แล้ว transaction ที่ "ต่างจากคนอื่น"  
จะถูก isolate ได้เร็วกว่า = anomaly score สูง

**ทำไมเพิ่ม IQR?**  
Isolation Forest อาจ flag รายการบ่อยแต่ยอดน้อย  
IQR rule (`Q3 + 1.5×IQR`) ช่วย filter ว่าต้องยอดสูงจริง ๆ ถึงจะ anomaly

## รองรับ CSV Format

KBank Cloud Pocket (`report_xxx-x-x659-6_YYYY-MM-DD_YYYY-MM-DD.csv`)

| Column | ใช้งาน |
|---|---|
| Txn | ยอดเงิน (ติดลบ = รายจ่าย) |
| Date | วันที่ |
| Type | ประเภท (Payment, Transfer Withdraw) |
| Note | ชื่อร้าน/รายละเอียด |
| Memo | บันทึกเพิ่มเติม |
| Category | หมวดหมู่จากธนาคาร |

## วิธีใช้

1. เปิด `finscan_v2.html` ใน Chrome / Safari
2. ลาก CSV วางในกล่อง upload
3. ดู dashboard ได้ทันที

## Tech Stack

- **Isolation Forest** — ported to vanilla JavaScript
- **Chart.js 4** — bar, doughnut, stacked charts
- **PapaParse** — CSV parsing
- ไม่มี backend, ไม่มี server, ไม่มี framework

## ไฟล์

```
finscan_v2.html              # Web app หลัก
mock_kbank_transactions.csv  # ข้อมูล mock สำหรับทดสอบ (ม.ค.–มี.ค. 2025)
```

---

*โปรเจกต์นี้สร้างเป็น Computer Science term project · ข้อมูลใน mock CSV เป็นข้อมูลสมมติ*
