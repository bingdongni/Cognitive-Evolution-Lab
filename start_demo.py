#!/usr/bin/env python3
"""
认知进化实验室 - 快速使用指南
演示脚本和示例数据的使用说明

作者: bingdongni
版本: v1.0.0
"""

def print_welcome():
    """打印欢迎信息和快速开始指南"""
    print("""
🧠 认知进化实验室 - 演示与示例数据 🧠
=================================================

欢迎使用认知进化实验室！这里提供了完整的演示脚本和示例数据，
帮助您快速体验和测试六种核心认知能力。

📁 快速文件导航:
├── scripts/demo.py              # 完整功能演示脚本
├── scripts/quick_start.py        # 快速入门演示
├── examples/                     # 配置文件目录
│   ├── basic_cognition_config.yaml
│   ├── creativity_training_config.yaml
│   └── evolution_experiment_config.yaml
├── data/examples/                # 示例数据目录
│   ├── cognitive_test_data.yaml
│   ├── memory_data.yaml
│   ├── reasoning_cases.yaml
│   └── creativity_tasks.yaml
└── tests/test_basic.py           # 基础功能测试

🚀 立即开始:

1️⃣  快速体验 (推荐新手):
   python scripts/quick_start.py

2️⃣  完整演示 (体验所有功能):
   python scripts/demo.py --mode full

3️⃣  特定认知能力测试:
   python scripts/demo.py --mode memory      # 记忆测试
   python scripts/demo.py --mode reasoning   # 推理测试
   python scripts/demo.py --mode creativity  # 创造力测试

4️⃣  基础功能测试:
   python tests/test_basic.py

💡 认知能力测试包括:
   🧠 记忆系统    - 情景、语义、工作、程序记忆
   🧩 推理能力    - 演绎、归纳、溯因、类比推理  
   🎨 创造力      - 发散思维、收敛思维、创意解决
   👁️ 观察力      - 视觉模式、异常检测、多尺度处理
   🎯 注意力      - 选择性、持续性、分散性注意
   🌟 想象力      - 情景、因果、时间、创意想象
   🧬 进化能力    - 协同进化、文化进化、环境共演化

📊 演示特色:
   ✅ 完整的中文界面和注释
   ✅ 实时的认知能力评分
   ✅ 详细的测试结果分析
   ✅ 可自定义的配置选项
   ✅ 完整的错误处理机制

🛠️ 高级使用:

自定义配置:
   python scripts/demo.py --config examples/creativity_training_config.yaml

保存结果到指定目录:
   python scripts/quick_start.py --output ./my_results

运行详细测试:
   python tests/test_basic.py --verbose

查看特定测试类型:
   python scripts/demo.py --help

📖 详细文档:
   - examples/README.md          - 配置文件说明
   - data/examples/README.md     - 数据文件说明
   - config/                     - 项目配置文件

🎯 测试场景:

🏃‍♂️ 快速体验场景 (5-10分钟):
   python scripts/quick_start.py --scenario basic

🎨 创造力训练场景 (15-20分钟):
   python scripts/demo.py --mode creativity

🧬 进化实验场景 (20-30分钟):
   python scripts/demo.py --mode evolution

🧠 完整认知测试场景 (30-45分钟):
   python scripts/demo.py --mode full

⚙️ 系统要求:
   - Python 3.7+
   - PyTorch (可选，用于深度学习模块)
   - 基础依赖已包含在 requirements.txt

🔧 故障排除:

问题: 导入模块失败
解决: 确保在项目根目录运行，或添加项目路径

问题: 演示运行过慢
解决: 使用基础配置:
     python scripts/demo.py --config examples/basic_cognition_config.yaml

问题: 内存不足
解决: 降低配置中的模型维度:
     - embed_dim: 128 (原512)
     - hidden_dim: 256 (原768)

问题: 需要GPU加速
解决: 系统会自动检测并使用可用设备

📞 技术支持:
   - 检查 config/ 目录下的配置文件
   - 查看各脚本中的详细注释
   - 运行测试脚本验证环境

开始您的认知探索之旅吧! 🚀
    """)

def show_menu():
    """显示交互式菜单"""
    while True:
        print("\n" + "="*60)
        print("🧠 认知进化实验室 - 演示菜单")
        print("="*60)
        print("1. 🏃‍♂️ 运行快速入门演示")
        print("2. 🎯 运行完整认知能力演示") 
        print("3. 🧠 测试特定认知能力")
        print("4. 🧬 测试进化功能")
        print("5. 🧪 运行基础功能测试")
        print("6. 📖 查看使用说明")
        print("7. 🚪 退出")
        print("="*60)
        
        choice = input("请选择操作 (1-7): ").strip()
        
        if choice == "1":
            print("\n🚀 启动快速入门演示...")
            import subprocess
            subprocess.run(["python", "scripts/quick_start.py"])
            
        elif choice == "2":
            print("\n🎯 启动完整认知能力演示...")
            import subprocess
            subprocess.run(["python", "scripts/demo.py", "--mode", "full"])
            
        elif choice == "3":
            print("\n🧠 选择要测试的认知能力:")
            print("1. 记忆系统 (memory)")
            print("2. 推理能力 (reasoning)")
            print("3. 创造力 (creativity)")
            print("4. 观察力 (observation)")
            print("5. 注意力 (attention)")
            print("6. 想象力 (imagination)")
            
            ability_choice = input("请输入能力编号 (1-6): ").strip()
            ability_map = {
                "1": "memory",
                "2": "reasoning", 
                "3": "creativity",
                "4": "observation",
                "5": "attention",
                "6": "imagination"
            }
            
            if ability_choice in ability_map:
                print(f"\n🧠 启动{ability_map[ability_choice]}能力测试...")
                import subprocess
                subprocess.run(["python", "scripts/demo.py", "--mode", ability_map[ability_choice]])
            else:
                print("❌ 无效选择")
                
        elif choice == "4":
            print("\n🧬 启动进化功能演示...")
            import subprocess
            subprocess.run(["python", "scripts/demo.py", "--mode", "evolution"])
            
        elif choice == "5":
            print("\n🧪 启动基础功能测试...")
            import subprocess
            subprocess.run(["python", "tests/test_basic.py"])
            
        elif choice == "6":
            print_welcome()
            
        elif choice == "7":
            print("\n👋 感谢使用认知进化实验室！")
            break
            
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    print_welcome()
    
    # 询问是否进入交互式菜单
    choice = input("\n是否进入交互式菜单? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes', '是', '1']:
        show_menu()
    else:
        print("\n💡 您可以随时运行以下命令开始体验:")
        print("   python scripts/quick_start.py        # 快速入门")
        print("   python scripts/demo.py --mode full   # 完整演示")
        print("   python tests/test_basic.py           # 功能测试")
        print("\n🎯 开始您的认知探索之旅吧!")