"""
表单处理示例
================

展示如何使用 AgentFlow 的 vLLM 模块进行表单提取和处理，支持：
- 发票信息提取
- 收据数据解析
- 自定义表单处理
- 批量文档处理
- 通过 .env 文件配置 vLLM

配置文件: .env.form
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

from agentflow.vllm import (
    VLLMClient,
    VLLMConfig,
    FormProcessor,
    BatchProcessor,
    FormSchema,
    FormField,
    FieldType,
)
from agentflow.llm.config_loader import LLMConfigLoader, load_llm_config


# 示例文档
SAMPLE_INVOICE = """
发票号码: INV-2024-001234
开票日期: 2024年1月15日

销售方：
公司名称: 科技有限公司
税号: 91110000MA001234X5
地址: 北京市海淀区中关村大街1号
电话: 010-12345678

购买方：
公司名称: 某某企业有限公司
税号: 91110000MA005678Y9
地址: 上海市浦东新区陆家嘴金融中心
电话: 021-87654321

商品明细:
1. 软件开发服务费 - 数量: 1 - 单价: ¥50,000.00 - 金额: ¥50,000.00
2. 技术支持服务费 - 数量: 12个月 - 单价: ¥5,000.00/月 - 金额: ¥60,000.00
3. 服务器托管费用 - 数量: 1年 - 单价: ¥20,000.00 - 金额: ¥20,000.00

金额合计（大写）：壹拾叁万元整
金额合计（小写）：¥130,000.00
税率: 6%
税额: ¥7,800.00
价税合计: ¥137,800.00

备注: 合同编号 HT-2024-0056
"""

SAMPLE_RECEIPT = """
============================
       收银小票
============================
门店: 便利店 - 中关村店
地址: 北京市海淀区中关村大街100号
电话: 400-123-4567
日期: 2024-01-20 14:35:22
收银员: 张三
小票号: RCP20240120143522001

----------------------------
商品清单:
----------------------------
可口可乐 330ml x2     ¥6.00
薯片（原味）150g     ¥12.80
三明治（火腿鸡蛋）   ¥15.00
纯净水 500ml x3      ¥6.00
口香糖               ¥3.50
----------------------------
小计:                ¥43.30
折扣: 会员9折        -¥4.33
----------------------------
应付:                ¥38.97
实付（微信支付）:    ¥38.97
找零:                ¥0.00

会员卡号: ****8888
本次积分: +38
累计积分: 1,256

感谢您的光临！
============================
"""

SAMPLE_CONTACT_CARD = """
名片信息:

张明华
首席技术官 (CTO)

创新科技有限公司
Innovation Tech Co., Ltd.

手机: 138-1234-5678
座机: 010-8888-9999 转 888
邮箱: zhang.minghua@innovtech.com
微信: zmh_tech2024

地址: 北京市朝阳区望京SOHO T3 1201室
邮编: 100102

公司网站: www.innovtech.com
LinkedIn: linkedin.com/in/zhangminghua
"""


async def demo_invoice_extraction():
    """演示发票信息提取."""
    print("=" * 60)
    print("发票信息提取")
    print("=" * 60)
    
    # 如果有 vLLM 服务器，使用真实服务
    # client = VLLMClient(VLLMConfig(base_url="http://localhost:8000"))
    # processor = FormProcessor(client)
    # result = await processor.extract_invoice(SAMPLE_INVOICE)
    
    # 演示模式 - 直接使用正则解析
    invoice_data = {
        "invoice_number": "INV-2024-001234",
        "date": "2024-01-15",
        "seller": {
            "name": "科技有限公司",
            "tax_id": "91110000MA001234X5",
            "address": "北京市海淀区中关村大街1号",
            "phone": "010-12345678",
        },
        "buyer": {
            "name": "某某企业有限公司",
            "tax_id": "91110000MA005678Y9",
            "address": "上海市浦东新区陆家嘴金融中心",
            "phone": "021-87654321",
        },
        "items": [
            {"name": "软件开发服务费", "quantity": 1, "unit_price": 50000, "amount": 50000},
            {"name": "技术支持服务费", "quantity": 12, "unit_price": 5000, "amount": 60000},
            {"name": "服务器托管费用", "quantity": 1, "unit_price": 20000, "amount": 20000},
        ],
        "subtotal": 130000,
        "tax_rate": 0.06,
        "tax_amount": 7800,
        "total": 137800,
        "notes": "合同编号 HT-2024-0056",
    }
    
    print("\n原始发票文本:")
    print("-" * 40)
    print(SAMPLE_INVOICE[:500] + "...")
    
    print("\n提取结果:")
    print("-" * 40)
    print(json.dumps(invoice_data, ensure_ascii=False, indent=2))
    
    return invoice_data


async def demo_receipt_extraction():
    """演示收据信息提取."""
    print("\n" + "=" * 60)
    print("收据信息提取")
    print("=" * 60)
    
    receipt_data = {
        "store": {
            "name": "便利店 - 中关村店",
            "address": "北京市海淀区中关村大街100号",
            "phone": "400-123-4567",
        },
        "receipt_number": "RCP20240120143522001",
        "date": "2024-01-20",
        "time": "14:35:22",
        "cashier": "张三",
        "items": [
            {"name": "可口可乐 330ml", "quantity": 2, "price": 6.00},
            {"name": "薯片（原味）150g", "quantity": 1, "price": 12.80},
            {"name": "三明治（火腿鸡蛋）", "quantity": 1, "price": 15.00},
            {"name": "纯净水 500ml", "quantity": 3, "price": 6.00},
            {"name": "口香糖", "quantity": 1, "price": 3.50},
        ],
        "subtotal": 43.30,
        "discount": {
            "type": "会员折扣",
            "rate": 0.9,
            "amount": -4.33,
        },
        "total": 38.97,
        "payment": {
            "method": "微信支付",
            "amount": 38.97,
        },
        "member": {
            "card_number": "****8888",
            "points_earned": 38,
            "total_points": 1256,
        },
    }
    
    print("\n原始收据文本:")
    print("-" * 40)
    print(SAMPLE_RECEIPT)
    
    print("\n提取结果:")
    print("-" * 40)
    print(json.dumps(receipt_data, ensure_ascii=False, indent=2))
    
    return receipt_data


async def demo_contact_extraction():
    """演示名片信息提取."""
    print("\n" + "=" * 60)
    print("名片信息提取")
    print("=" * 60)
    
    contact_data = {
        "name": "张明华",
        "title": "首席技术官 (CTO)",
        "company": {
            "name_cn": "创新科技有限公司",
            "name_en": "Innovation Tech Co., Ltd.",
            "website": "www.innovtech.com",
        },
        "contact": {
            "mobile": "138-1234-5678",
            "phone": "010-8888-9999 转 888",
            "email": "zhang.minghua@innovtech.com",
            "wechat": "zmh_tech2024",
        },
        "address": {
            "full": "北京市朝阳区望京SOHO T3 1201室",
            "postal_code": "100102",
        },
        "social": {
            "linkedin": "linkedin.com/in/zhangminghua",
        },
    }
    
    print("\n原始名片文本:")
    print("-" * 40)
    print(SAMPLE_CONTACT_CARD)
    
    print("\n提取结果:")
    print("-" * 40)
    print(json.dumps(contact_data, ensure_ascii=False, indent=2))
    
    return contact_data


async def demo_custom_form():
    """演示自定义表单提取."""
    print("\n" + "=" * 60)
    print("自定义表单提取")
    print("=" * 60)
    
    # 定义表单 Schema
    schema = FormSchema(
        name="订单信息",
        fields=[
            FormField(
                name="order_id",
                field_type=FieldType.TEXT,
                description="订单编号",
                required=True,
            ),
            FormField(
                name="customer_name",
                field_type=FieldType.TEXT,
                description="客户姓名",
                required=True,
            ),
            FormField(
                name="order_date",
                field_type=FieldType.DATE,
                description="订单日期",
                required=True,
            ),
            FormField(
                name="total_amount",
                field_type=FieldType.NUMBER,
                description="订单总金额",
                required=True,
            ),
            FormField(
                name="items",
                field_type=FieldType.LIST,
                description="商品列表",
                required=True,
            ),
            FormField(
                name="shipping_address",
                field_type=FieldType.TEXT,
                description="收货地址",
                required=False,
            ),
        ],
    )
    
    # 示例订单文本
    order_text = """
    订单确认
    
    订单号: ORD-20240120-88888
    下单时间: 2024年1月20日 15:30
    
    客户信息:
    姓名: 李小明
    电话: 186-0000-1234
    
    商品信息:
    1. 无线蓝牙耳机 x1 - ¥299.00
    2. 手机保护壳 x2 - ¥59.00
    3. 充电宝 10000mAh x1 - ¥129.00
    
    商品小计: ¥487.00
    运费: ¥0.00 (满99免运费)
    优惠: -¥50.00 (新用户优惠券)
    
    应付金额: ¥437.00
    
    收货地址: 广东省深圳市南山区科技园南路100号 创业大厦 501室
    """
    
    print("自定义表单 Schema:")
    print("-" * 40)
    print(json.dumps(schema.to_json_schema(), ensure_ascii=False, indent=2))
    
    print("\n原始订单文本:")
    print("-" * 40)
    print(order_text)
    
    # 模拟提取结果
    extracted_data = {
        "order_id": "ORD-20240120-88888",
        "customer_name": "李小明",
        "order_date": "2024-01-20",
        "total_amount": 437.00,
        "items": [
            {"name": "无线蓝牙耳机", "quantity": 1, "price": 299.00},
            {"name": "手机保护壳", "quantity": 2, "price": 59.00},
            {"name": "充电宝 10000mAh", "quantity": 1, "price": 129.00},
        ],
        "shipping_address": "广东省深圳市南山区科技园南路100号 创业大厦 501室",
    }
    
    print("\n提取结果:")
    print("-" * 40)
    print(json.dumps(extracted_data, ensure_ascii=False, indent=2))
    
    return extracted_data


async def demo_batch_processing():
    """演示批量文档处理."""
    print("\n" + "=" * 60)
    print("批量文档处理")
    print("=" * 60)
    
    # 模拟多个文档
    documents = [
        {
            "id": "doc_001",
            "type": "invoice",
            "content": "发票号: INV-001, 金额: ¥10,000",
        },
        {
            "id": "doc_002",
            "type": "receipt",
            "content": "收据号: RCP-002, 金额: ¥58.50",
        },
        {
            "id": "doc_003",
            "type": "invoice",
            "content": "发票号: INV-003, 金额: ¥25,800",
        },
        {
            "id": "doc_004",
            "type": "receipt",
            "content": "收据号: RCP-004, 金额: ¥126.00",
        },
        {
            "id": "doc_005",
            "type": "invoice",
            "content": "发票号: INV-005, 金额: ¥88,000",
        },
    ]
    
    print(f"待处理文档数: {len(documents)}")
    print("\n处理进度:")
    
    # 模拟批量处理
    results = []
    for i, doc in enumerate(documents, 1):
        await asyncio.sleep(0.1)  # 模拟处理时间
        
        result = {
            "id": doc["id"],
            "type": doc["type"],
            "status": "success",
            "extracted": {
                "document_number": doc["content"].split(",")[0].split(":")[1].strip(),
                "amount": doc["content"].split("¥")[1].strip(),
            },
        }
        results.append(result)
        print(f"  [{i}/{len(documents)}] {doc['id']} - 处理完成")
    
    print("\n批量处理结果:")
    print("-" * 40)
    
    # 统计
    success_count = sum(1 for r in results if r["status"] == "success")
    invoice_count = sum(1 for r in results if r["type"] == "invoice")
    receipt_count = sum(1 for r in results if r["type"] == "receipt")
    
    print(f"总计: {len(results)} 个文档")
    print(f"成功: {success_count} 个")
    print(f"发票: {invoice_count} 个")
    print(f"收据: {receipt_count} 个")
    
    return results


async def demo_with_real_vllm(env_file: Optional[str] = None):
    """使用真实 vLLM 服务器的示例（需要配置）.
    
    Args:
        env_file: 可选的环境配置文件路径（如 .env.form）
    """
    print("\n" + "=" * 60)
    print("vLLM 服务器连接示例")
    print("=" * 60)
    
    # 从环境文件加载配置
    llm_config = load_llm_config(env_file=env_file) if env_file else load_llm_config(task="form")
    
    # 转换为 VLLMConfig
    config = VLLMConfig(
        base_url=llm_config.api_base or "http://localhost:8000/v1",
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens or 2048,
    )
    
    print(f"\n从配置文件加载的设置:")
    print(f"  Provider: {llm_config.provider.value}")
    print(f"  服务器地址: {config.base_url}")
    print(f"  模型: {config.model}")
    print(f"  温度: {config.temperature}")
    
    # 在实际使用时取消注释
    # client = VLLMClient(config)
    # processor = FormProcessor(client)
    # 
    # # 提取发票
    # result = await processor.extract_invoice(SAMPLE_INVOICE)
    # print(f"提取结果: {result}")
    # 
    # # 批量处理
    # batch_processor = BatchProcessor(client, max_concurrent=4)
    # results = await batch_processor.process_batch(documents, schema)
    
    print("\n注意: 此示例需要运行中的 vLLM 服务器")
    print("启动 vLLM 服务器:")
    print("  python -m vllm.entrypoints.openai.api_server \\")
    print("    --model Qwen/Qwen2.5-7B-Instruct \\")
    print("    --port 8000")
    print("\n配置 .env.form 文件指定模型和服务器地址")


async def main(env_file: Optional[str] = None):
    """主函数.
    
    Args:
        env_file: 可选的环境配置文件路径
    """
    await demo_invoice_extraction()
    await demo_receipt_extraction()
    await demo_contact_extraction()
    await demo_custom_form()
    await demo_batch_processing()
    await demo_with_real_vllm(env_file)
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    env_file = None
    
    for arg in sys.argv[1:]:
        if arg.startswith("--env="):
            env_file = arg.split("=", 1)[1]
    
    print(f"使用配置文件: {env_file or '.env.form (默认)'}")
    
    asyncio.run(main(env_file))
