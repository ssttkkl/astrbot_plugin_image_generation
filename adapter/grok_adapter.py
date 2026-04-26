from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.types import GenerationRequest, ImageCapability


class GrokAdapter(BaseImageAdapter):
    """Grok（xAI）图像生成适配器。"""

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return (
            ImageCapability.TEXT_TO_IMAGE
            | ImageCapability.ASPECT_RATIO
            | ImageCapability.RESOLUTION
            | ImageCapability.IMAGE_TO_IMAGE
        )

    # generate() 方法由基类提供，使用模板方法模式

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        start_time = time.time()
        prefix = self._get_log_prefix(request.task_id)

        payload = self._build_payload(request)
        session = self._get_session()

        if request.images:
            end_point = "/images/edits"
        else:
            end_point = "/images/generations"

        if not self.base_url:
            url = f"https://api.x.ai/v1{end_point}"
        else:
            # 考虑到 main.py 会清理掉 /v1，这里统一加上
            url = f"{self.base_url.rstrip('/')}/v1{end_point}"

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as resp:
                duration = time.time() - start_time
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"{prefix} API 错误 ({resp.status}, 耗时: {duration:.2f}s): {error_text}"
                    )
                    return None, f"API 错误 ({resp.status})"

                data = await resp.json()
                logger.info(f"{prefix} 生成成功 (耗时: {duration:.2f}s)")
                return await self._extract_images(data)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {e}")
            return None, str(e)

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建请求载荷。"""

        accept_ratio = [
            "auto",
            "1:1",
            "16:9",
            "9:16",
            "4:3",
            "3:4",
            "3:2",
            "2:3",
            "1:2",
            "2:1",
            "19.5:9",
            "9:19.5",
            "20:9",
            "9:20",
        ]
        accept_resolution = ["1k", "2k"]

        ratio = "auto"
        if request.aspect_ratio in accept_ratio:
            ratio = request.aspect_ratio
        if request.aspect_ratio == "自动":
            ratio = "auto"

        resolution = "2k"
        if request.resolution.lower() in accept_resolution:
            resolution = request.resolution.lower()

        images_ref = []
        for image in request.images:
            b64_data = base64.b64encode(image.data).decode("utf-8")
            images_ref.append(
                {
                    "type": "image_url",
                    "url": f"data:{image.mime_type};base64,{b64_data}",
                }
            )

        payload: dict[str, Any] = {
            "model": self.model or "grok-imagine-image",
            "prompt": request.prompt,
            "aspect_ratio": ratio,
            "resolution": resolution,
            "response_format": "b64_json",
        }

        if len(images_ref) > 0:
            payload.update({"images": images_ref})

        return payload

    async def _extract_images(
        self, response: dict
    ) -> tuple[list[bytes] | None, str | None]:
        """从响应中提取图片数据。"""
        if "data" not in response:
            return None, "响应中未找到 data 字段"

        images = []
        for item in response["data"]:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                # 如果返回的是 URL，需要下载（虽然我们请求的是 b64_json）
                async with self._get_session().get(
                    item["url"], proxy=self.proxy, timeout=self._get_download_timeout()
                ) as resp:
                    if resp.status == 200:
                        images.append(await resp.read())

        if not images:
            return None, "未找到有效的图片数据"

        return images, None
