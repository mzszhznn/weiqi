// 导入TensorFlow.js库，并创建一个神经网络模型
import * as tf from "@tensorflow/tfjs-node";
var model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [19, 19, 3],
  filters: 64,
  kernelSize: [3, 3],
  activation: "relu"
}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({
  units: 361,
  activation: "softmax"
}));

// 定义一个函数，用来根据当前轮到哪一方下棋，默认为黑方  0表示空白，1表示黑子，2表示白子
var turn = 1;

// 创建一个19*19的二维数组，用来存储棋盘上每个点的状态
// 0表示空白，1表示黑子，2表示白子
var board = new Array(19);
for (var i = 0; i < board.length; i++) {
  board[i] = new Array(19);
  for (var j = 0; j < board[i].length; j++) {
    board[i][j] = 0;
  }
}

// 定义一个函数，用来记录并输出每一步棋
function record(color, x, y) {
  // 将颜色转换为中文字符
  var colorStr = color == 1 ? "黑" : "白";
  // 将坐标转换为字母数字组合
  var xStr = String.fromCharCode(65 + x);
  var yStr = String(19 - y);
  // 拼接成棋谱格式，并打印到控制台
  var recordStr = colorStr + " " + xStr + yStr;
  console.log(recordStr);
}


// 定义一个函数，用来根据当前棋盘状态预测下一步最佳落子位置
function predict(board) {
  // 将二维数组转换为三维张量，并扩展第一维度为批量大小为1 
  var input = tf.tensor(board).expandDims(0);

  // 使用模型进行预测，并获取最大概率对应的索引 
  var output = model.predict(input).argMax(-1);

  // 将索引转换为坐标，并返回 
  var index = output.dataSync()[0];
  var x = index % 19;
  var y = Math.floor(index / 19);
  return [x, y];
}


// 定义一个函数，用来执行一次对弈过程  
function play() {
  // 初始化棋盘状态和轮次 
  board = new Array(19);
  for (var i = 0; i < board.length; i++) {
    board[i] = new Array(19);
    for (var j = 0; j < board[i].length; j++) {
      board[i][j] = 0;
    }
  }
  turn = 1;
  // 循环直到游戏结束 
  while (true) {
    // 如果当前轮到黑方下棋，则执行蒙特卡洛树搜索，并获取最优动作 
    if (turn == 1) {
      var [x, y] = mcts(board, turn);
    } else {
      // 如果当前轮到白方下棋，则使用模型预测最优动作 
      var [x, y] = predict(board);
    }
    // 如果该位置已经有子，则游戏结束 
    if (hasStone(x, y)) { break; }
    // 将子落在该位置 
    board[x][y] = turn;
    // 记录该步棋 
    record(turn, x, y);
    // 如果该子没有气，则将其吃掉 
    if (!hasLiberty(x, y)) {
      // 获取该子周围四个方向的坐标 
      var neighbors = getNeighbors(x, y);
      // 遍历每个邻居 
      for (var [i, j] of neighbors) {
        // 如果邻居和当前位置颜色相同，并且没有气，则将其吃掉 
        if (board[i][j] == turn && !hasLiberty(i, j)) {
          board[i][j] = 0;
        }
      }
    }
    // 切换轮次 
    turn = oppositeColor(x, y);
  }
}

// 定义一个函数，用来判断棋盘上某个位置是否为空
function isEmpty(x, y, newBoard) {
  if (newBoard) return newBoard[x][y] == 0;
  return board[x][y] == 0;
}

// 定义一个函数，用来判断棋盘上某个位置是否有子
function hasStone(x, y) {
  return board[x][y] != 0;
}

// 定义一个函数，用来判断棋盘上某个位置是否有黑子
function isBlack(x, y) {
  return board[x][y] == 1;
}

// 定义一个函数，用来判断棋盘上某个位置是否有白子
function isWhite(x, y) {
  return board[x][y] == 2;
}

// 定义一个函数，用来获取棋盘上某个位置相反颜色的子
function oppositeColor(x, y) {
  if (isBlack(x, y)) {
    return 2;
  } else if (isWhite(x, y)) {
    return 1;
  } else {
    return 0;
  }
}

// 定义一个函数，用来判断棋盘上某个位置是否在边界内
function inBound(x, y) {
  return x >= 0 && x < 19 && y >= 0 && y < 19;
}

// 定义一个函数，用来获取棋盘上某个位置周围四个方向的坐标
function getNeighbors(x, y) {
  var neighbors = [];
  if (inBound(x - 1, y)) {
    neighbors.push([x - 1, y]);
  }
  if (inBound(x + 1, y)) {
    neighbors.push([x + 1, y]);
  }
  if (inBound(x, y - 1)) {
    neighbors.push([x, y - 1]);
  }
  if (inBound(x, y + 1)) {
    neighbors.push([x, y + 1]);
  }
  return neighbors;
}

// 定义一个函数，用来判断棋盘上某个位置的子是否有气 
function hasLiberty(x, y, newBoard) {
  let _board = newBoard || board;
  // 如果该位置为空，直接返回false 
  if (isEmpty(x, y, _board)) { return false; }
  // 获取该位置的颜色 
  var color = _board[x][y];
  // 创建一个队列，用来存储待检查的坐标 
  var queue = [];
  // 创建一个集合，用来存储已检查过的坐标
  var visited = new Set();
  // 将该位置加入队列和集合 
  queue.push([x, y]);
  visited.add(x + "," + y);
  // 循环直到队列为空 
  while (queue.length > 0) {
    // 取出队首元素 
    var [i, j] = queue.shift();
    // 获取其周围四个方向的坐标 
    var neighbors = getNeighbors(i, j);
    // 遍历每个邻居 
    for (var [k, l] of neighbors) {
      // 如果邻居为空，说明有气，直接返回true 
      if (isEmpty(k, l, _board)) { return true; }
      // 如果邻居和当前位置颜色相同，并且没有被检查过，将其加入队列和集合
      if (_board[k][l] == color && !visited.has(k + "," + l)) {
        queue.push([k, l]);
        visited.add(k + "," + l);
      }
    }
  }
  // 如果所有邻居都没有气，返回false 
  return false;
}


// 定义一个函数，用来执行一次蒙特卡洛树搜索，并返回最优动作和价值 
function mcts(board, turn) {
  // 创建一个根节点，存储当前棋盘状态、轮次、访问次数、累积价值和子节点 
  var root = { board: board, turn: turn, visitCount: 0, valueSum: 0, children: [] };
  // 循环搜索指定次数 
  for (var i = 0; i < 1000; i++) {
    // 从根节点开始选择最优子节点，直到达到叶子节点或未扩展的节点
    var node = root;
    while (node.children.length > 0 && !isTerminal(node.board)) {
      node = selectBestChild(node);
    }
    // 如果该节点未扩展，则扩展所有可能的动作，并随机选择一个子节点
    if (node.children.length == 0 && !isTerminal(node.board)) {
      expand(node);
      node = selectRandomChild(node);
    }
    // 模拟该子节点到游戏结束，并获取最终结果
    var result = simulate(node);
    // 回溯更新所有经过的节点的访问次数和累积价值 
    backpropagate(node, result);
  }
  // 在根节点的所有子节点中选择访问次数最多的动作和平均价值作为最优动作和价值，并返回 
  var bestAction = null;
  var bestValue = -Infinity;
  for (var child of root.children) {
    var value = child.valueSum / child.visitCount;
    if (child.visitCount > bestValue) {
      bestAction = child.action;
      bestValue = value;
    }
  }
  // 返回最优动作和价值 
  return [bestAction, bestValue];
}

// 定义一个函数，用来选择最优子节点，使用上限置信区间（UCB）公式 
function selectBestChild(node) {
  // 初始化最优子节点、最大得分和探索常数 
  var bestChild = null;
  var maxScore = -Infinity;
  var c = 1.4;
  // 遍历每个子节点 
  for (var child of node.children) {
    // 计算该子节点的平均价值和探索因子 
    var value = child.valueSum / child.visitCount;
    var exploration = Math.sqrt(Math.log(node.visitCount) / child.visitCount);
    // 计算该子节点的UCB得分 
    var score = value + c * exploration;
    // 如果该得分大于当前最大得分，则更新最优子节点和最大得分
    if (score > maxScore) {
      bestChild = child;
      maxScore = score;
    }
  }
  // 返回最优子节点 
  return bestChild;
}
// 定义一个函数，用来判断棋盘是否达到终局状态 
function isTerminal(board) {
  // 如果棋盘上没有空位，或者有一方的子被全部吃掉，则返回true 
  if (isEmptyCount(board) == 0 || getStoneCount(board, 1) == 0 || getStoneCount(board, 2) == 0) {
    return true;
  }
  // 否则返回false 
  return false;
}
// 完成isEmptyCount方法
// 定义一个函数，用来计算棋盘上空位的数目，并返回它
function isEmptyCount(board) {
  // 定义一个变量，用来存储空位的数目 
  var count = 0;
  // 遍历棋盘上的每个位置 
  for (var i = 0; i < board.length; i++) {
    for (var j = 0; j < board[i].length; j++) {
      // 如果该位置为空，则增加计数器 
      if (isEmpty(i, j)) {
        count++;
      }
    }
  }
  // 返回计数器的值 
  return count;
}

// 完成getStoneCount方法
// 定义一个函数，用来计算棋盘上某一种颜色的子的数目，并返回它
function getStoneCount(board, color) {
  // 定义一个变量，用来存储子的数目 
  var count = 0;
  // 遍历棋盘上的每个位置 
  for (var i = 0; i < board.length; i++) {
    for (var j = 0; j < board[i].length; j++) {
      // 如果该位置的颜色与参数相同，则增加计数器 
      if (board[i][j] == color) {
        count++;
      }
    }
  }
  // 返回计数器的值 
  return count;
}
// 定义一个函数，用来扩展所有可能的动作，并创建子节点 
function expand(node) {
  // 获取当前节点的棋盘状态和轮次 
  var board = node.board;
  var turn = node.turn;
  // 遍历棋盘上的每个空位 
  for (var i = 0; i < board.length; i++) {
    for (var j = 0; j < board[i].length; j++) {
      if (isEmpty(i, j)) {
        // 尝试在该空位落子，并判断是否合法 
        var newBoard = copyBoard(board);
        newBoard[i][j] = turn;
        if (isValidMove(newBoard, i, j)) {
          // 如果合法，则创建一个子节点，存储新的棋盘状态、轮次、动作、访问次数、累积价值和子节点，并将其加入当前节点的子节点列表中 
          var newTurn = turn == 1 ? 2 : 1;
          var action = [i, j];
          var child = { board: newBoard, turn: newTurn, action: action, visitCount: 0, valueSum: 0, children: [] };
          node.children.push(child);
        }
      }
    }
  }
}

// 定义一个函数，用来复制棋盘状态 
function copyBoard(board) {
  // 创建一个新的二维数组
  var newBoard = [];
  // 遍历原棋盘上的每个位置
  for (var i = 0; i < board.length; i++) {
    // 创建一个新的一维数组
    var newRow = [];
    for (var j = 0; j < board[i].length; j++) {
      // 将原棋盘上的每个位置的值复制到新的一维数组中 
      newRow.push(board[i][j]);
    }
    // 将新的一维数组加入到新的二维数组中
    newBoard.push(newRow);
  }
  // 返回新的二维数组
  return newBoard;
}


function isValidMove(board, i, j, tryColor) {
  // 获取该位置的颜色 
  var color = board[i][j];
  // 判断该位置是否为空，如果不为空，则返回false 
  if (color != 0) {
    return false;
  }
  // 尝试在该位置放置自己的颜色，并判断是否有气 
  board[i][j] = tryColor;
  var result = hasLiberty(i, j, board);
  // 还原棋盘状态，并返回结果 
  board[i][j] = 0;
  return result;
}

// 定义一个函数，用来从当前节点的子节点列表中随机选择一个子节点，并返回它 
function selectRandomChild(node) {
  // 获取当前节点的子节点列表 
  var children = node.children;
  // 如果没有子节点，则返回null 
  if (children.length == 0) { return null; }
  // 随机生成一个索引，范围在0到子节点列表的长度减一之间 
  var index = Math.floor(Math.random() * children.length);
  // 返回对应索引的子节点 
  return children[index];
}

// 定义一个函数，用来模拟从当前节点开始的一次随机游戏，并返回游戏的结果（胜负平）
function simulate(node) {
  // 获取当前节点的棋盘状态和轮次    
  var board = copyBoard(node.board);
  var turn = node.turn;
  // 定义一个变量，用来存储游戏是否结束的标志
  var gameOver = false;
  //双方累计没有可以放置的位置时累计,超过2也就是双方都没有
  var invalidCount = 0;
  // 定义一个循环，直到游戏结束
  while (!gameOver) {
    // 定义一个数组，用来存储所有合法的动作
    var validMoves = [];
    // 遍历棋盘上的每个空位 
    for (var i = 0; i < board.length; i++) {
      for (var j = 0; j < board[i].length; j++) {
        if (isEmpty(i, j)) {
          // 尝试在该空位落子，并判断是否合法  
          if (isValidMove(board, i, j, turn)) {
            // 如果合法，则将该动作加入到合法动作数组中
            validMoves.push([i, j]);
          }
        }
      }
    }
    // 判断是否有合法的动作，如果没有，则切换轮次，并继续循环
    if (validMoves.length == 0) {
      invalidCount++;
      if (invalidCount >= 2) {
        gameOver = true;
        break;
      } else {
        turn = turn == 1 ? 2 : 1;
        continue;
      }
    }
    invalidCount = 0;
    // 如果有合法的动作，则从中随机选择一个，并更新棋盘状态和轮次
    var index = Math.floor(Math.random() * validMoves.length);
    var move = validMoves[index];
    board[move[0]][move[1]] = turn;
    turn = turn == 1 ? 2 : 1;
    // 判断是否游戏结束，即棋盘上没有空位或者双方都没有合法的动作
    gameOver = isGameOver(board);
  }
  // 如果游戏结束，则计算黑白双方的棋子数目，并返回结果（1表示黑胜，-1表示白胜，0表示平局）
  var blackCount = countColor(board, 1);
  var whiteCount = countColor(board, 2);
  if (blackCount > whiteCount) {
    return 1;
  } else if (blackCount < whiteCount) {
    return -1;
  } else {
    return 0;
  }
}

// 完成backpropagate方法
// 定义一个函数，用来从当前节点开始，沿着父节点链向上更新每个节点的访问次数和平均得分
function backpropagate(node, result) {
  // 定义一个变量，用来存储当前节点
  var current = node;
  // 定义一个循环，直到当前节点为空
  while (current != null) {
    // 更新当前节点的访问次数和平均得分，根据结果和轮次调整符号
    current.visitCount++;
    current.valueSum += result * (current.turn == 1 ? 1 : -1);
    // 将当前节点更新为其父节点
    current = current.parent;
  }
}
// 完成isGameOver方法
// 定义一个函数，用来判断棋盘上的游戏是否结束，即棋盘上没有空位或者双方都没有合法的动作
function isGameOver(board) {
  // 遍历棋盘上的每个位置 
  for (var i = 0; i < board.length; i++) {
    for (var j = 0; j < board[i].length; j++) {
      // 如果有空位，则返回false
      if (isEmpty(i, j)) {
        return false;
      }
      // 如果有合法的动作，则返回false
      if (isValidMove(board, i, j)) {
        return false;
      }
    }
  }
  // 如果没有空位或者合法的动作，则返回true
  return true;
}

// 完成countColor方法
// 定义一个函数，用来计算棋盘上某一种颜色的棋子数目，并返回它
function countColor(board, color) {
  // 定义一个变量，用来存储棋子数目 
  var count = 0;
  // 遍历棋盘上的每个位置 
  for (var i = 0; i < board.length; i++) {
    for (var j = 0; j < board[i].length; j++) {
      // 如果该位置的颜色与参数相同，则增加计数器 
      if (board[i][j] == color) {
        count++;
      }
    }
  }
  // 返回计数器的值 
  return count;
}


play();