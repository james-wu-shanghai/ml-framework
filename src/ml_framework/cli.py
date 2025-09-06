"""
命令行接口

提供ML Framework的命令行工具
"""

import click
import os
import sys
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_framework import MLFramework


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """ML Framework 命令行工具"""
    pass


@cli.command()
@click.option('--data', '-d', required=True, help='数据文件路径')
@click.option('--target', '-t', required=True, help='目标列名')
@click.option('--model', '-m', default='random_forest', help='模型名称')
@click.option('--task-type', default='classification', help='任务类型')
@click.option('--config', '-c', help='配置文件路径')
@click.option('--output', '-o', default='models/trained_model.joblib', help='模型输出路径')
def train(data, target, model, task_type, config, output):
    """训练模型"""
    click.echo(f"开始训练模型...")
    click.echo(f"数据: {data}")
    click.echo(f"目标: {target}")
    click.echo(f"模型: {model}")
    click.echo(f"任务类型: {task_type}")
    
    try:
        # 初始化框架
        framework = MLFramework(config_path=config)
        
        # 训练流程
        framework.load_data(data, target_column=target)
        framework.set_task_type(task_type)
        framework.preprocess_data()
        framework.select_model(model)
        framework.train()
        
        # 评估
        results = framework.evaluate()
        click.echo("\n训练结果:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                click.echo(f"  {metric}: {value:.4f}")
        
        # 保存模型
        os.makedirs(os.path.dirname(output), exist_ok=True)
        framework.save_model(output)
        click.echo(f"\n模型已保存到: {output}")
        
    except Exception as e:
        click.echo(f"训练失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', required=True, help='模型文件路径')
@click.option('--data', '-d', required=True, help='测试数据路径')
@click.option('--target', '-t', required=True, help='目标列名')
@click.option('--task-type', default='classification', help='任务类型')
def evaluate(model, data, target, task_type):
    """评估模型"""
    click.echo(f"评估模型: {model}")
    click.echo(f"测试数据: {data}")
    
    try:
        # 初始化框架
        framework = MLFramework()
        
        # 加载模型和数据
        framework.load_model(model)
        framework.load_data(data, target_column=target)
        framework.set_task_type(task_type)
        
        # 评估
        results = framework.evaluate()
        
        click.echo("\n评估结果:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                click.echo(f"  {metric}: {value:.4f}")
                
    except Exception as e:
        click.echo(f"评估失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', required=True, help='模型文件路径')
@click.option('--data', '-d', required=True, help='预测数据路径')
@click.option('--output', '-o', default='predictions.csv', help='预测结果输出路径')
def predict(model, data, output):
    """使用模型进行预测"""
    click.echo(f"使用模型: {model}")
    click.echo(f"预测数据: {data}")
    
    try:
        import pandas as pd
        
        # 初始化框架
        framework = MLFramework()
        
        # 加载模型
        framework.load_model(model)
        
        # 加载数据
        data_df = pd.read_csv(data)
        
        # 预测
        predictions = framework.predict(data_df)
        
        # 保存结果
        result_df = data_df.copy()
        result_df['prediction'] = predictions
        result_df.to_csv(output, index=False)
        
        click.echo(f"\n预测完成，结果已保存到: {output}")
        click.echo(f"预测样本数: {len(predictions)}")
        
    except Exception as e:
        click.echo(f"预测失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data', '-d', required=True, help='数据文件路径')
@click.option('--target', '-t', required=True, help='目标列名')
@click.option('--task-type', default='classification', help='任务类型')
@click.option('--config', '-c', help='配置文件路径')
def auto(data, target, task_type, config):
    """自动机器学习"""
    click.echo("运行自动机器学习...")
    
    try:
        # 初始化框架
        framework = MLFramework(config_path=config)
        
        # 自动ML
        results = framework.auto_ml(
            data_path=data,
            target_column=target,
            task_type=task_type
        )
        
        click.echo("\n自动ML结果:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                click.echo(f"  {metric}: {value:.4f}")
                
    except Exception as e:
        click.echo(f"自动ML失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def init():
    """初始化新项目"""
    click.echo("初始化ML Framework项目...")
    
    # 创建目录结构
    dirs = ['data', 'models', 'logs', 'plots', 'configs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        click.echo(f"创建目录: {dir_name}/")
    
    # 创建配置文件
    config_content = """# ML Framework 配置文件
data:
  batch_size: 32
  test_size: 0.2
  random_state: 42

models:
  random_forest:
    n_estimators: 100
    random_state: 42

training:
  validation_split: 0.2

logging:
  level: "INFO"
  file: "logs/ml_framework.log"
"""
    
    with open('configs/config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    click.echo("创建配置文件: configs/config.yaml")
    click.echo("\n项目初始化完成！")
    click.echo("开始使用: ml-framework train --data your_data.csv --target target_column")


def main():
    """主函数"""
    cli()


if __name__ == '__main__':
    main()