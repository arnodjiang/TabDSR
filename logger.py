import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, name=None, log_path="logs"):
        """指定保存日志的文件路径，日志级别，以及调用文件
        将日志存入到指定的文件中
  
		Examples:
			‍‍‍```python
				try:
				    xxx
				except Exception as e:
					mylogger.getlog.error(e)  
					mylogger.getlog.error("异常堆栈跟踪:\n%s", traceback.format_exc())
			‍‍‍```
        """
        self.name=name if name else __name__
        os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)
        # 创建一个logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
        # 创建一个handler，用于写入日志文件
            rq = datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"{self.name}.log"
            log_name =os.path.join(os.getcwd(), "logs", rq)
            fh = logging.FileHandler(log_name, mode='a', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
        # 再创建一个handler，用于输出到控制台
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
        # 定义handler的输出格式
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)d: [\n%(message)s\n]')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
        # 给logger添加handler
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        self.loglevel = {
	        "debug": self.logger.debug,
	        "info": self.logger.info,
	        "warning": self.logger.warning,
	        "error": self.logger.error,
	        "critical": self.logger.critical
	    }

    @property
    def log(self):
        return self.logger.debug

    @property
    def debug(self):
        return self.logger.debug
  
    @property
    def info(self):
        return self.logger.info
  
    @property
    def warning(self):
        return self.logger.warning
  
    @property
    def error(self):
        return self.logger.error
  
    @property
    def critical(self):
        return self.logger.critical
  
    def printlog(self):
        return self.loglevel.values()

    def demo(self):
        try:
            self.printlog()
        except Exception as e:
            self.logger.debug("测试debug")
            self.logger.critical("异常堆栈跟踪:\n%s", traceback.format_exc())