package com.detector.batch.core

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

object ObjectMapperProvider {
  private val _mapper = {
    try {
      val mapper = new ObjectMapper()
      mapper.registerModule(DefaultScalaModule)
      Some(mapper)
    } catch {
      case exception: Exception => ;
        None
    }
  }

  def getMapper: Option[ObjectMapper] = _mapper
}
