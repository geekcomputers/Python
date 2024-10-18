from typing import Dict, List
        

class Dictionary:

    def __init__(self):
        self.node = {}

    def add_word(self, word: str) -> None:
        node = self.node
        for ltr in word:
            if ltr not in node:
                node[ltr] = {}
            node = node[ltr]
        node["is_word"] = True

    def word_exists(self, word: str) -> bool:
        node = self.node
        for ltr in word:
            if ltr not in node:
                return False
            node = node[ltr]
        return "is_word" in node

    def list_words_from_node(self, node: Dict, spelling: str) -> None:
        if "is_word" in node:
            self.words_list.append(spelling)
            return
        for ltr in node:
            self.list_words_from_node(node[ltr], spelling+ltr)

    def print_all_words_in_dictionary(self) -> List[str]:
        node = self.node
        self.words_list = []
        self.list_words_from_node(node, "")
        return self.words_list

    def suggest_words_starting_with(self, prefix: str) -> List[str]:
        node = self.node
        for ltr in prefix:
            if ltr not in node:
                return False
            node = node[ltr]
        self.words_list = []
        self.list_words_from_node(node, prefix)
        return self.words_list

    


# Your Dictionary object will be instantiated and called as such:
obj = Dictionary()
obj.add_word("word")
obj.add_word("woke")
obj.add_word("happy")

param_2 = obj.word_exists("word")
param_3 = obj.suggest_words_starting_with("wo")

print(param_2)
print(param_3)
print(obj.print_all_words_in_dictionary())