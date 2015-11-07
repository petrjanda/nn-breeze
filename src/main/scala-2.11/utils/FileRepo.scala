package utils

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import scala.util.Try

class FileRepo(dir:String) {
  def save[T](obj: T, path: String) = {
    val os = new ObjectOutputStream(new FileOutputStream(dir + path))
    try {
      os.writeObject(obj)
    } finally {
      os.close()
    }
  }

  def load[T](path: String): Try[T] = Try {
    val is = new ObjectInputStream(new FileInputStream(dir + path))
    try {
      is.readObject().asInstanceOf[T]
    } finally {
      is.close()
    }
  }
}
